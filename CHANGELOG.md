# Changelog

All notable changes to the Event Camera Viewer project are documented in this file.

## [Unreleased] - 2025-11-04

### Added

#### Core Features
- **Dynamic Bias Range Detection**: Camera biases now query hardware-specific ranges at startup
  - Automatically adjusts slider ranges based on camera capabilities
  - Initializes with camera's current values instead of hardcoded defaults
  - Reset button uses middle of actual range
  - Prevents invalid bias value crashes

- **Hardware Monitoring Panel**: Real-time sensor statistics display
  - Temperature monitoring (°C)
  - Pixel dead time display (microseconds)
  - Automatic capability detection (disables unsupported features)
  - Graceful handling of cameras without illumination sensors

- **Region of Interest (ROI) Control**: Hardware-level event filtering
  - Interactive X/Y position sliders (0 to image dimensions)
  - Width/Height size sliders with real-time updates
  - Enable/disable checkbox for ROI filtering
  - **Visual Feedback**: Green rectangle overlay showing ROI window position
  - **Crop to ROI View**: Optional zoomed view showing only ROI region
  - Real-time slider updates (ROI window moves as you drag)
  - Auto-apply when ROI enabled (no button clicks required)

- **Event Rate Controller (ERC)**: Bandwidth management
  - Enable/disable checkbox
  - Target event rate slider (events/second)
  - Min/max rate display based on camera capabilities
  - Prevents USB bandwidth saturation

- **Anti-Flicker Filter Module**: Artificial lighting noise removal
  - Enable/disable checkbox
  - Filter mode dropdown (BAND_STOP / BAND_PASS)
  - Frequency band control (low/high frequency sliders)
  - Duty cycle adjustment
  - **Quick Preset Buttons**: One-click common frequencies
    - 50Hz preset (Europe/Asia AC lighting)
    - 60Hz preset (North America AC lighting)
    - 100Hz preset (second harmonic)
    - 120Hz preset (second harmonic)
  - Range clamping for preset buttons
  - Error handling for out-of-range frequencies

- **Quick Launcher Script**: `run_event_camera.bat`
  - Automatically navigates to build directory
  - Checks for executable existence
  - Launches application with proper working directory
  - User-friendly error messages

#### Documentation
- **AVAILABLE_CAMERA_FEATURES.md**: Comprehensive SDK feature reference
  - Documents all Metavision SDK facilities (I_ROI, I_ErcModule, etc.)
  - Usage examples with code snippets
  - Priority recommendations for GUI implementation
  - Thread safety and error handling notes

- **Updated README.md**: Complete feature documentation
  - All advanced camera features documented
  - Detailed usage instructions for each feature
  - Troubleshooting section with common issues
  - Technical notes on thread safety and performance

### Changed

#### Configuration
- **Renamed Configuration File**: `tracking_config.ini` → `event_config.ini`
  - Removed all non-camera parameters (stereo, tracking, simulation, etc.)
  - Focused solely on event camera hardware settings
  - Updated all code references to new filename
  - Auto-copy to build directory via CMakeLists.txt

- **Default Config Path**: Updated from `../../../tracking_config.ini` to `../../../event_config.ini`
  - Modified in `app_config.h` and `app_config.cpp`

#### API Updates
- **Fixed Metavision SDK API Compatibility**:
  - Changed `camera.get_facility<>()` → `camera.get_device().get_facility<>()`
  - Applied consistently across all facility access
  - Ensures compatibility with current SDK version

#### Thread Safety Improvements
- **Frame Generator Mutex Protection**:
  - Added `framegen_mutex` for thread-safe frame generator access
  - Protected event callback with mutex lock
  - Protected frame generator recreation with mutex lock
  - Prevents race conditions when adjusting accumulation time

- **ROI Visualization Thread Safety**:
  - Added `ROIVisualization` struct with dedicated mutex
  - Protected ROI overlay updates with mutex lock
  - Prevents race conditions during real-time slider updates

#### User Interface Improvements
- **Accumulation Time Control**:
  - Changed from immediate update to "Apply Settings" button
  - Prevents crashes during slider adjustment
  - Clear visual feedback for pending changes

- **Bias Controls**:
  - Removed hardcoded ranges (previously 0-255)
  - Sliders now use camera-specific min/max values
  - Display current hardware values on startup
  - Safe error handling for unsupported biases

- **ROI Controls**:
  - Removed "Apply ROI Window" button
  - Auto-apply when ROI enabled and sliders change
  - Real-time visual feedback as sliders move
  - Separate checkbox for "Crop View to ROI"

### Fixed

#### Critical Bug Fixes
- **Accumulation Slider Crash** (Race Condition):
  - Problem: UI thread destroying frame_gen while event thread using it
  - Solution: Mutex-protected frame generator access
  - Added null check in event callback
  - Changed to button-based apply instead of immediate update

- **Invalid Bias Value Crash** (Startup):
  - Problem: Hardcoded bias value (128) outside camera's range [-25, 23]
  - Solution: Query actual bias ranges using `get_bias_info()` and `get_bias_range()`
  - Initialize sliders with camera's current values
  - Dynamic range detection for all biases

- **Illumination Error Spam** (Hardware Compatibility):
  - Problem: Console flooded with "Failed to get illumination" errors
  - Solution: Capability checking at startup
  - Explicitly disable illumination: `monitoring_caps.has_illumination = false`
  - Only query supported features during runtime

- **Bias Query Exception** (Hardware Compatibility):
  - Problem: Some biases don't exist on all camera models
  - Solution: Wrapped all bias queries in try-catch blocks
  - Graceful error messages for missing biases
  - Application continues even if some biases unavailable

- **Anti-Flicker Preset Crash** (Range Validation):
  - Problem: Preset frequencies might be outside camera's supported range
  - Solution: Added range clamping using `std::max()`/`std::min()`
  - Wrapped preset button calls in try-catch blocks
  - Console logging for debugging

#### USB Connection Issues (User-Facing)
- **LIBUSB_ERROR_TIMEOUT**: Documented workaround
  - Close all instances of application
  - Unplug camera, wait 5 seconds, replug
  - Try different USB 3.0 port
  - Added to troubleshooting documentation

### Technical Improvements

#### Error Handling
- Capability checking for all hardware facilities
- Try-catch blocks around facility access
- Graceful degradation when features unavailable
- Detailed console logging for debugging

#### Code Organization
- Added comprehensive inline documentation
- Separated ROI visualization logic into dedicated struct
- Consistent mutex locking patterns
- Clear variable naming conventions

#### Build System
- CMake auto-copies `event_config.ini` to build directory
- Post-build custom command for config file
- Ensures config file always available at runtime

### Removed
- `tracking_config.ini` (replaced by `event_config.ini`)
- Hardcoded bias ranges (replaced by dynamic detection)
- Illumination sensor monitoring (not supported by hardware)
- "Apply ROI Window" button (replaced by auto-apply)

---

## Development Notes

### Hardware Tested
- CenturyArks SilkyEvCam HD
  - Bias range: -25 to 23 (not 0-255 as initially assumed)
  - Temperature monitoring: ✓ Supported
  - Illumination sensor: ✗ Not supported
  - ROI: ✓ Supported
  - ERC: ✓ Supported
  - Anti-Flicker: ✓ Supported

### Known Issues
- Illumination monitoring not available on current hardware (by design)
- Some bias parameters may not exist on all camera models
- USB timeout errors require manual camera reconnection

### Future Enhancements
Potential features for future implementation (see AVAILABLE_CAMERA_FEATURES.md):
- Event Trail Filter (TRAIL, STC_CUT_TRAIL, STC_KEEP_TRAIL modes)
- Digital Crop (similar to ROI but less flexible)
- Trigger In/Out (multi-camera synchronization)
- Save/Load camera presets
- Event recording to file
- Histogram/statistics overlays

---

## Migration Guide

### For Users Updating from Previous Version

1. **Configuration File Change**:
   - Old: `tracking_config.ini`
   - New: `event_config.ini`
   - Action: Rename your config file or let CMake copy the new default

2. **Bias Value Ranges**:
   - Old: 0-255 (hardcoded)
   - New: Camera-specific (e.g., -25 to 23)
   - Action: Bias values in config will be clamped to valid range automatically

3. **New Features Available**:
   - Hardware Monitoring panel
   - Region of Interest control with visualization
   - Event Rate Controller
   - Anti-Flicker filter with presets
   - Quick launcher script (`run_event_camera.bat`)

4. **Removed Features**:
   - None (all previous features retained)

### For Developers

1. **API Changes**:
   ```cpp
   // OLD:
   auto* facility = camera.get_facility<Metavision::I_LL_Biases>();

   // NEW:
   auto* facility = camera.get_device().get_facility<Metavision::I_LL_Biases>();
   ```

2. **Thread Safety**:
   - Always use `framegen_mutex` when accessing frame generator
   - Use `roi_viz.mutex` when accessing ROI visualization data
   - Check for null before using facilities in callbacks

3. **Adding New Features**:
   ```cpp
   // Template for adding new facility
   auto* new_facility = camera.get_device().get_facility<Metavision::I_NewFacility>();
   if (new_facility) {
       // Feature available, safe to use
       try {
           new_facility->some_operation();
       } catch (const std::exception& e) {
           std::cerr << "Error: " << e.what() << std::endl;
       }
   } else {
       // Feature not supported on this camera
       std::cout << "New facility not available" << std::endl;
   }
   ```

---

## Version History

### Unreleased (Current)
- All features listed above
- Based on Metavision SDK 4.x
- Tested on Windows 10/11 with Visual Studio 2022

### Previous (Pre-changelog)
- Basic camera connection
- Simple bias controls (0-255 range)
- Frame accumulation adjustment
- Basic ImGui interface
