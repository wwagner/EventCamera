# Binary Stream Implementation Plan

## Overview
Split 8-bit video datastream into two 1-bit streams (Down/Up) as early as possible to save processing power.

**Down = Range 3 [96-127]** (mid-range pixels)
**Up = Range 7 [224-255]** (bright pixels)

## Current Status (Committed: bf24313)

### What's Already Done
- ✅ Added `BinaryStreamMode` enum in `include/core/display_settings.h`
  - OFF = 0 (8-bit passthrough)
  - DOWN = 1 (Range 3 only)
  - UP = 2 (Range 7 only)
  - UP_DOWN = 3 (Both ranges combined)
- ✅ Added `set_binary_stream_mode()` and `get_binary_stream_mode()`
- ✅ Added UI dropdown in `src/ui/settings_panel.cpp` (line ~385)
- ✅ Added atomic storage `binary_stream_mode_` in DisplaySettings
- ✅ Grayscale mode infrastructure exists

### What Needs to Be Implemented

## Phase 1: Create Binary Stream Conversion Infrastructure

### 1.1 Add LUT Creation Helper Function
**Location:** `src/main.cpp`, near top after includes

```cpp
/**
 * Create lookup table for binary stream conversion
 * Maps pixel values in [range_lower, range_upper] to 255, all others to 0
 */
static cv::Mat create_binary_stream_lut(int range_lower, int range_upper) {
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; i++) {
        lut.at<uchar>(i) = (i >= range_lower && i <= range_upper) ? 255 : 0;
    }
    return lut;
}
```

### 1.2 Create Static Lookup Tables
**Location:** `src/main.cpp`, global scope

```cpp
// Binary stream lookup tables (initialized once for performance)
static cv::Mat lut_down;      // Range 3: [96-127] → 255, else → 0
static cv::Mat lut_up;         // Range 7: [224-255] → 255, else → 0
static cv::Mat lut_combined;   // Both ranges → 255, else → 0
static bool luts_initialized = false;

void initialize_binary_stream_luts() {
    if (luts_initialized) return;

    // Down stream: Range 3 [96-127]
    lut_down = create_binary_stream_lut(96, 127);

    // Up stream: Range 7 [224-255]
    lut_up = create_binary_stream_lut(224, 255);

    // Combined: Both ranges
    lut_combined = cv::Mat(1, 256, CV_8U);
    for (int i = 0; i < 256; i++) {
        bool in_down = (i >= 96 && i <= 127);
        bool in_up = (i >= 224 && i <= 255);
        lut_combined.at<uchar>(i) = (in_down || in_up) ? 255 : 0;
    }

    luts_initialized = true;
}
```

### 1.3 Create Binary Stream Conversion Function
**Location:** `src/main.cpp`, before frame callback functions

```cpp
/**
 * Apply binary stream mode conversion (early 1-bit conversion)
 * This is the FIRST processing step - happens before all other processing
 *
 * @param frame Input 8-bit frame (BGR or grayscale)
 * @param mode Binary stream mode (OFF/DOWN/UP/UP_DOWN)
 * @return Processed frame (single-channel 1-bit if mode != OFF)
 */
cv::Mat apply_binary_stream_mode(const cv::Mat& frame,
                                  core::DisplaySettings::BinaryStreamMode mode) {
    using Mode = core::DisplaySettings::BinaryStreamMode;

    if (mode == Mode::OFF) {
        return frame;  // Pass through unchanged (8-bit)
    }

    initialize_binary_stream_luts();

    // Convert to single-channel grayscale if needed
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    // Apply LUT for ultra-fast 1-bit conversion
    cv::Mat binary_1bit;
    switch (mode) {
        case Mode::DOWN:    cv::LUT(gray, lut_down, binary_1bit); break;
        case Mode::UP:      cv::LUT(gray, lut_up, binary_1bit); break;
        case Mode::UP_DOWN: cv::LUT(gray, lut_combined, binary_1bit); break;
        default: binary_1bit = gray; break;
    }

    // Return as single-channel (most efficient)
    return binary_1bit;
}
```

## Phase 2: Update Frame Callbacks

### 2.1 Main Camera Callbacks (2 cameras)
**Location:** `src/main.cpp`, line ~575, function `setup_camera_frame_callbacks`

**Find this code:**
```cpp
frame_gen->set_output_callback([camera_index](const Metavision::timestamp ts, cv::Mat& frame) {
    if (frame.empty() || !app_state) return;

    // Debug: Print frame info once
    static bool printed_once = false;
    if (!printed_once) {
        // ...
    }

    // Convert to grayscale if requested (BGR -> GRAY -> BGR for display compatibility)
    if (app_state->display_settings().get_grayscale_mode() && frame.channels() == 3) {
        // ...
    }
```

**Replace with:**
```cpp
frame_gen->set_output_callback([camera_index](const Metavision::timestamp ts, cv::Mat& frame) {
    if (frame.empty() || !app_state) return;

    // *** STEP 1: EARLY BINARY STREAM CONVERSION ***
    // This happens FIRST - convert 8-bit to 1-bit based on range selection
    auto stream_mode = app_state->display_settings().get_binary_stream_mode();
    frame = apply_binary_stream_mode(frame, stream_mode);

    // *** STEP 2: Optional grayscale (now works on already-converted 1-bit if stream mode is on) ***
    // NOTE: If stream mode is ON, frame is already single-channel, so this only applies if OFF
    if (app_state->display_settings().get_grayscale_mode() && frame.channels() == 3) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);
    }

    // *** STEP 3: Convert to BGR for GPU upload if needed ***
    if (frame.channels() == 1) {
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    }

    // Rate limit display updates to target FPS
    auto now = std::chrono::steady_clock::now();
    // ... rest of callback unchanged
});
```

### 2.2 GA Frame Generation Callback
**Location:** `src/main.cpp`, function `evaluate_genome_fitness`, line ~254

**Find this code:**
```cpp
app_state->camera_state().frame_generator()->set_output_callback([](const Metavision::timestamp ts, cv::Mat& frame) {
    if (frame.empty() || !app_state) return;

    // If GA is capturing, store in GA buffer (unprocessed)
    if (ga_state.capturing_for_ga) {
        ga_state.ga_frame_buffer.store_frame(frame.clone());
    }

    // Apply display processing (for display only, not for GA capture)
    cv::Mat display_frame = frame.clone();

    // Convert to grayscale if requested
```

**Replace with:**
```cpp
app_state->camera_state().frame_generator()->set_output_callback([](const Metavision::timestamp ts, cv::Mat& frame) {
    if (frame.empty() || !app_state) return;

    // Store RAW frame for GA if capturing (before any processing)
    if (ga_state.capturing_for_ga) {
        ga_state.ga_frame_buffer.store_frame(frame.clone());
    }

    // Apply display processing (for display only, not for GA capture)
    cv::Mat display_frame = frame.clone();

    // *** STEP 1: EARLY BINARY STREAM CONVERSION ***
    auto stream_mode = app_state->display_settings().get_binary_stream_mode();
    display_frame = apply_binary_stream_mode(display_frame, stream_mode);

    // *** STEP 2: Optional grayscale ***
    if (app_state->display_settings().get_grayscale_mode() && display_frame.channels() == 3) {
        cv::Mat gray;
        cv::cvtColor(display_frame, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(gray, display_frame, cv::COLOR_GRAY2BGR);
    }

    // *** STEP 3: Convert to BGR if single-channel ***
    if (display_frame.channels() == 1) {
        cv::cvtColor(display_frame, display_frame, cv::COLOR_GRAY2BGR);
    }

    // Rate limit display updates
```

### 2.3 GA Fitness Evaluation Processing
**Location:** `src/main.cpp`, function `evaluate_genome_fitness`, line ~360

**Find this code:**
```cpp
// Apply display processing to frames if requested
if (config.ga_settings().use_processed_pixels && app_state) {
    for (auto& frame : captured_frames) {
        // Apply grayscale conversion if enabled
        if (app_state->display_settings().get_grayscale_mode() && frame.channels() == 3) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);
        }

        // Apply binary threshold if enabled
        if (app_state->display_settings().get_binary_mode() && !frame.empty()) {
```

**Replace with:**
```cpp
// Apply display processing to frames if requested
if (config.ga_settings().use_processed_pixels && app_state) {
    for (auto& frame : captured_frames) {
        // *** STEP 1: EARLY BINARY STREAM CONVERSION ***
        auto stream_mode = app_state->display_settings().get_binary_stream_mode();
        frame = apply_binary_stream_mode(frame, stream_mode);

        // *** STEP 2: Optional grayscale ***
        if (app_state->display_settings().get_grayscale_mode() && frame.channels() == 3) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);
        }

        // *** STEP 3: Ensure BGR for downstream processing ***
        if (frame.channels() == 1) {
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }
    }
}
```

## Phase 3: Remove Old Binary Threshold Code

### 3.1 Delete from display_settings.h
**REMOVE these declarations:**
- ❌ `void set_binary_mode(bool enabled);`
- ❌ `bool get_binary_mode() const;`
- ❌ `void set_binary_threshold(int threshold);`
- ❌ `int get_binary_threshold() const;`
- ❌ `std::atomic<bool> binary_mode_{false};`
- ❌ `std::atomic<int> binary_threshold_{...};`

**KEEP these:**
- ✅ `enum class BinaryStreamMode { ... }`
- ✅ `void set_binary_stream_mode(BinaryStreamMode mode);`
- ✅ `BinaryStreamMode get_binary_stream_mode() const;`
- ✅ `std::atomic<int> binary_stream_mode_{0};`

### 3.2 Delete from display_settings.cpp
**Already done** - old methods were replaced with new stream mode methods

### 3.3 Delete from settings_panel.cpp
**Already done** - UI was replaced with stream mode dropdown

### 3.4 Clean Up main.cpp
**REMOVE these sections:**
1. Old binary conversion code in main camera callback (~line 651-684)
2. Old binary conversion code in GA callback (~line 272-286)
3. Old binary conversion code in GA fitness (~line 370-410)
4. Any debug output related to old binary threshold

**Search for and DELETE:**
- Any code with `get_binary_mode()`
- Any code with `get_binary_threshold()`
- Any `cv::inRange()` calls for binary filtering
- Debug code printing "Frame values - Min: Max: Mean:"

## Phase 4: Performance & Technical Details

### Why This is Fast
1. **LUT operations**: O(1) per pixel lookup, ~1-2 CPU cycles
2. **Single-channel output**: 1/3 memory bandwidth vs BGR (900KB vs 2.7MB)
3. **Early conversion**: Only happens once at source, not repeated
4. **Downstream benefits**: All processing works on simple 0/255 values

### Memory Flow
```
8-bit BGR frame (3 channels, 1280x720 = 2.7 MB)
    ↓
[apply_binary_stream_mode]
    ↓
Single-channel grayscale (1 channel = 900 KB)
    ↓
LUT applied (still 900 KB, but only 0 or 255 values) ← 1-bit data
    ↓
[Convert to BGR for GPU upload]
    ↓
BGR output (2.7 MB, but values are 0 or 255 only)
```

### Key Technical Points
- **LUT size**: 256 bytes (fits in L1 cache)
- **LUT lookup**: Single array access per pixel
- **No branches**: LUT eliminates if/else per pixel
- **Cache friendly**: Sequential memory access

## Phase 5: Testing Checklist

- [ ] Build succeeds
- [ ] Mode OFF: Display works exactly as before (8-bit passthrough)
- [ ] Mode DOWN: Only pixels [96-127] show as white, rest black
- [ ] Mode UP: Only pixels [224-255] show as white, rest black
- [ ] Mode UP_DOWN: Both ranges show as white, rest black
- [ ] Grayscale mode still works with all stream modes
- [ ] GA optimization works with "Use Processed Pixels" enabled
- [ ] Frame combine (add images) still works
- [ ] No crashes on mode switching
- [ ] Performance improvement observable (smoother display)

## Implementation Order

1. ✅ Commit current work (done: bf24313)
2. Add LUT functions and initialization (Phase 1.1-1.2)
3. Add apply_binary_stream_mode() function (Phase 1.3)
4. Update main camera callbacks (Phase 2.1)
5. Update GA frame callback (Phase 2.2)
6. Update GA fitness processing (Phase 2.3)
7. Remove old binary code (Phase 3)
8. Test thoroughly (Phase 5)
9. Commit final implementation

## Notes

- **Down stream** = Range 3 = [96-127] = Mid-range pixels
- **Up stream** = Range 7 = [224-255] = Bright pixels
- Binary conversion happens **BEFORE** grayscale mode
- Output is single-channel until GPU upload (most efficient)
- All old binary threshold code will be completely removed
