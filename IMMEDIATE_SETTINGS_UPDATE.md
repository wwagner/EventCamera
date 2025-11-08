# Immediate Settings Application Implementation

## Overview
Updated the settings panel to apply most settings immediately when changed, eliminating the need for an "Apply Settings" button. Settings that require a camera restart are now clearly marked in a different color.

## Changes Implemented

### 1. Immediate Bias Application ✓

**Modified File**: `src/ui/settings_panel.cpp` - `render_bias_controls()`

**Changes**:
- Added `apply_bias_immediately()` helper lambda that calls BiasManager
- Updated all 6 bias sliders to apply changes immediately when moved:
  - `bias_diff` - Event detection threshold
  - `bias_diff_on` - ON event threshold
  - `bias_diff_off` - OFF event threshold
  - `bias_refr` - Refractory period
  - `bias_fo` - Photoreceptor follower
  - `bias_hpf` - High-pass filter

**Before**:
```cpp
if (ImGui::SliderFloat("bias_diff", &slider_diff, 0.0f, 100.0f, "%.0f%%")) {
    cam_settings.bias_diff = exp_to_bias(slider_diff, range.min, range.max);
    settings_changed_ = true;  // Just flag for later
}
```

**After**:
```cpp
if (ImGui::SliderFloat("bias_diff", &slider_diff, 0.0f, 100.0f, "%.0f%%")) {
    cam_settings.bias_diff = exp_to_bias(slider_diff, range.min, range.max);
    apply_bias_immediately("bias_diff", cam_settings.bias_diff);  // Apply NOW to all cameras
}
```

**Result**: When you move a bias slider, the change is **instantly** applied to **both cameras** via BiasManager.

### 2. Restart-Required Settings Marked ✓

**Modified File**: `src/ui/settings_panel.cpp` - `render_frame_generation()`

**Changes**:
- Accumulation time slider rendered in **orange/yellow color** (`ImVec4(1.0f, 0.8f, 0.2f, 1.0f)`)
- Shows **"Need Restart - Reconnect camera to apply"** warning text when changed
- Uses `ImGui::PushStyleColor()` / `ImGui::PopStyleColor()` to colorize the slider

**Code**:
```cpp
// Mark restart-required settings in orange/yellow
ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
if (ImGui::SliderFloat("Accumulation (s)", &cam_settings.accumulation_time_s,
                      0.001f, 0.1f, "%.3f")) {
    accumulation_changed = true;
    settings_changed_ = true;
}
ImGui::PopStyleColor();

if (cam_settings.accumulation_time_s != previous_accumulation) {
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Need Restart - Reconnect camera to apply");
}
```

**Result**: User can see at a glance which settings require restart - they're in **orange/yellow** color.

### 3. Apply Settings Button Removed ✓

**Modified File**: `src/ui/settings_panel.cpp` - `render_apply_button()`

**Before** (43 lines):
- Showed "Settings changed!" warning
- "Apply Settings" button that manually applied biases to all cameras
- Complex logic to iterate cameras and apply settings

**After** (10 lines):
```cpp
void SettingsPanel::render_apply_button() {
    ImGui::Spacing();
    ImGui::Separator();

    // Note: Most settings now apply immediately when changed
    // Only accumulation time requires restart

    ImGui::Spacing();

    if (ImGui::Button("Reset to Defaults", ImVec2(-1, 0))) {
        // Reset bias manager and apply to all cameras immediately
        bias_mgr_.reset_to_defaults();
        bias_mgr_.apply_to_camera();

        // Reset accumulation time (requires restart to take effect)
        config_.camera_settings().accumulation_time_s = 0.01f;

        std::cout << "All settings reset to defaults" << std::endl;
    }
}
```

**Result**:
- No more "Apply Settings" button
- Only "Reset to Defaults" button remains
- Reset also applies immediately via BiasManager

### 4. Settings That Apply Immediately ✓

The following settings now apply instantly when changed:

| Setting | Type | Cameras Affected | How Applied |
|---------|------|------------------|-------------|
| **Analog Biases** | | | |
| - bias_diff | Slider/Input | Both | BiasManager |
| - bias_diff_on | Slider/Input | Both | BiasManager |
| - bias_diff_off | Slider/Input | Both | BiasManager |
| - bias_refr | Slider/Input | Both | BiasManager |
| - bias_fo | Slider/Input | Both | BiasManager |
| - bias_hpf | Slider/Input | Both | BiasManager |
| **Digital Features** | | | |
| - ERC (Event Rate Control) | Checkbox/Slider | Both | FeatureManager |
| - Anti-Flicker | Checkbox/Sliders | Both | FeatureManager |
| - Trail Filter | Checkbox/Combo/Slider | Both | FeatureManager |
| - ROI (Region of Interest) | Checkbox/Sliders | Both | FeatureManager |
| **Display Settings** | | | |
| - Target FPS | Slider | N/A | UI only |
| - Frame Subtraction | Checkbox | N/A | UI only |

### 5. Settings That Require Restart ⚠️

These settings are marked in **orange/yellow** and show "Need Restart" warning:

| Setting | Why Restart Needed | Color |
|---------|-------------------|-------|
| Accumulation Time | Frame generator initialized at camera connection | Orange/Yellow |

To apply these changes:
1. Adjust the setting (saves to config)
2. Click "Disconnect & Reconnect Camera" button at top
3. Camera will reconnect with new settings

## User Experience Improvements

### Before This Update:
1. User moves bias slider
2. UI shows "Settings changed!" in yellow
3. User clicks "Apply Settings" button
4. Settings applied to both cameras
5. Console output confirms application

**Total steps**: 3 clicks

### After This Update:
1. User moves bias slider
2. Settings **instantly** applied to both cameras
3. Console output confirms immediate application

**Total steps**: 1 action

**Time saved**: ~2-3 seconds per settings change

## Technical Details

### BiasManager Integration
The `apply_bias_immediately()` lambda uses BiasManager's multi-camera support:

```cpp
auto apply_bias_immediately = [&](const std::string& name, int value) {
    bias_mgr_.set_bias(name, value);      // Update BiasManager state
    bias_mgr_.apply_to_camera();          // Apply to ALL cameras
};
```

BiasManager automatically:
- Validates the value against hardware limits
- Clamps to valid range
- Applies to all registered cameras (Camera 0 and Camera 1)
- Outputs console confirmation

### Thread Safety
- BiasManager uses mutex protection for multi-camera operations
- Settings updates are synchronous (blocking until complete)
- No race conditions between UI thread and camera threads

### FeatureManager
Digital features already applied immediately through FeatureManager:
- Each feature (ERC, Anti-Flicker, Trail Filter, ROI) has its own module
- Changes are applied to Camera 0 via FeatureManager
- `apply_digital_features_to_all_cameras()` syncs to other cameras
- This behavior was already in place, not modified

## Console Output Examples

### Bias Change (Immediate):
```
BiasManager: Applying biases to 2 camera(s)...
  Camera 0:
    bias_hpf=105
  Camera 1:
    bias_hpf=105
BiasManager: Biases applied successfully to all cameras
```

### Reset to Defaults:
```
BiasManager: Resetting biases to defaults (middle of range)
BiasManager: Applying biases to 2 camera(s)...
  Camera 0:
    bias_diff=0
    bias_refr=107
    bias_fo=10
    bias_hpf=60
  Camera 1:
    bias_diff=0
    bias_refr=107
    bias_fo=10
    bias_hpf=60
BiasManager: Biases applied successfully to all cameras
All settings reset to defaults
```

### Accumulation Time Change:
```
(No immediate console output - saved to config)
(After reconnecting cameras):
Frame accumulation time: 0.05s
```

## Files Modified

1. **src/ui/settings_panel.cpp**
   - `render_bias_controls()`: Added immediate application
   - `render_frame_generation()`: Added color/warning for restart settings
   - `render_apply_button()`: Removed "Apply Settings" button, simplified

## Testing Checklist

- [x] Bias sliders apply immediately to both cameras
- [x] Bias input boxes apply immediately to both cameras
- [x] Reset to Defaults applies immediately
- [x] Accumulation time shows in orange/yellow color
- [x] "Need Restart" warning appears for accumulation time
- [x] No "Apply Settings" button visible
- [x] Digital features still work (already immediate)
- [x] Console output confirms immediate application

## Benefits

1. **Faster Workflow**: No extra button clicks needed
2. **Clearer UX**: Obvious which settings need restart (color-coded)
3. **Immediate Feedback**: See effects instantly while adjusting
4. **Dual-Camera Sync**: Both cameras always in sync
5. **Simpler Code**: Removed ~30 lines of redundant application logic

## Future Enhancements

If needed, could add:
1. Undo/Redo for accidental slider moves
2. "Lock Settings" toggle to prevent accidental changes
3. Per-camera control toggle (currently all settings sync both cameras)
4. Settings presets/profiles for quick switching

For now, immediate application provides the best user experience for real-time camera tuning.
