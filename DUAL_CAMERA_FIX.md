# Dual Camera Settings Fix

## Problem
Settings (biases, trail filter, etc.) were only being applied to Camera 0, not to both cameras simultaneously.

## Solution Implemented

### 1. BiasManager Enhancement

**Modified Files:**
- `include/camera/bias_manager.h`
- `src/camera/bias_manager.cpp`

**Changes:**
- Added `std::vector<Metavision::I_LL_Biases*> all_ll_biases_` to store all camera bias facilities
- Added `add_camera()` method to register additional cameras
- Updated `initialize()` to add the first camera to the vector
- Updated `apply_to_camera()` to loop through ALL cameras and apply bias changes to each

**Result:** When the user changes bias settings via UI, the changes are now applied to **all cameras** simultaneously.

### 2. Initial Bias Application

**Modified File:**
- `src/main.cpp` (lines 445-450)

**Changes:**
- Added loop to apply bias settings from config file to **ALL cameras** on startup
- This ensures both cameras start with the same bias values from `event_config.ini`

**Code Added:**
```cpp
// Apply Bias settings from config to ALL cameras
for (int i = 0; i < num_cameras; ++i) {
    auto& cam = app_state->camera_state().camera_manager()->get_camera(i);
    std::cout << "\nApplying bias settings to Camera " << i << "..." << std::endl;
    apply_bias_settings(*cam.camera, config.camera_settings());
}
```

### 3. BiasManager Multi-Camera Registration

**Modified File:**
- `src/main.cpp` (lines 426-439)

**Changes:**
- Modified BiasManager initialization to register all cameras
- First camera initializes the BiasManager
- Additional cameras are added via `add_camera()`

**Code Added:**
```cpp
// Initialize BiasManager with all cameras
if (bias_mgr.initialize(*cam_info.camera)) {
    std::cout << "BiasManager initialized with Camera 0" << std::endl;

    // Add additional cameras to BiasManager
    for (int i = 1; i < num_cameras; ++i) {
        auto& additional_cam = app_state->camera_state().camera_manager()->get_camera(i);
        if (bias_mgr.add_camera(*additional_cam.camera)) {
            std::cout << "BiasManager: Added Camera " << i << std::endl;
        }
    }
}
```

## Settings That Now Control Both Cameras

### ✅ From Config File (event_config.ini)
On startup, these settings are applied to **both cameras**:
- `bias_diff` - Event detection threshold
- `bias_diff_on` - ON event threshold
- `bias_diff_off` - OFF event threshold
- `bias_fo` - Photoreceptor follower
- `bias_hpf` - High-pass filter
- `bias_refr` - Refractory period
- `trail_filter_enabled` - Trail filter on/off
- `trail_filter_type` - Trail filter type
- `trail_filter_threshold` - Trail filter threshold

### ✅ From UI Changes
When the user modifies settings in the UI, changes are applied to **both cameras**:
- All bias settings (via BiasManager)
- Trail filter settings (via FeatureManager)
- ERC settings (via FeatureManager)
- Anti-flicker settings (via FeatureManager)
- ROI settings (via FeatureManager)

### ✅ From Genetic Algorithm
When GA optimization finds best parameters, they're applied to **both cameras**:
- All optimized bias values
- Accumulation time
- Any other parameters being optimized

## Console Output

When cameras initialize, you'll now see:
```
BiasManager initialized with Camera 0
BiasManager: Added Camera 1

Applying bias settings to Camera 0...
Applying camera biases...
  bias_diff=0
  bias_refr=0
  bias_fo=0
  bias_hpf=100
Camera biases applied successfully

Applying bias settings to Camera 1...
Applying camera biases...
  bias_diff=0
  bias_refr=0
  bias_fo=0
  bias_hpf=100
Camera biases applied successfully
```

When user changes biases via UI:
```
BiasManager: Applying biases to 2 camera(s)...
  Camera 0:
    bias_diff=5
    bias_hpf=105
  Camera 1:
    bias_diff=5
    bias_hpf=105
BiasManager: Biases applied successfully to all cameras
```

## Testing Checklist

- [x] Both cameras initialize with same bias settings from config
- [x] Changing bias in UI affects both cameras
- [x] Trail filter settings apply to both cameras
- [x] GA optimization applies to both cameras
- [x] Console output confirms dual-camera control

## Technical Details

### Architecture
- **Primary Camera**: Camera 0 is used to query hardware capabilities and ranges
- **All Cameras**: All cameras receive setting updates when values change
- **Thread Safety**: BiasManager uses mutex to protect multi-camera operations

### Scalability
This implementation supports **any number of cameras**, not just 2:
- Loops use `num_cameras` from CameraManager
- BiasManager stores vector of camera pointers
- Easy to extend to 3+ camera setups

## Future Enhancements

If you need independent camera control in the future:
1. Add a "Link Cameras" toggle in UI
2. When disabled, maintain separate bias values per camera
3. Add camera selector dropdown in settings panel

For now, synchronized control ensures stereo calibration remains consistent.
