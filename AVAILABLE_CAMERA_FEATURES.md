# Available Event Camera Features

This document lists the camera features available in the Metavision SDK that could be exposed in the Event Camera Viewer GUI.

## Currently Exposed Features ✓

### Camera Biases (via I_LL_Biases)
- **bias_diff** - Event detection threshold (hardware-specific range)
- **bias_refr** - Refractory period (hardware-specific range)
- **bias_fo** - Photoreceptor follower (hardware-specific range)
- **bias_hpf** - High-pass filter (hardware-specific range)
- **bias_pr** - Pixel photoreceptor (hardware-specific range)
- **Dynamic range detection** - Queries camera-specific min/max values

### Frame Generation
- **accumulation_time_s** - Event accumulation period (0.001-0.1s)

### Hardware Monitoring (via I_Monitoring)
- **Temperature** - Sensor temperature monitoring (°C)
- **Pixel Dead Time** - Refractory period display (microseconds)
- **Capability Detection** - Automatic detection of supported features

### Region of Interest (via I_ROI)
- **ROI Window** - Rectangular region control (X, Y, Width, Height)
- **Visual Overlay** - Green rectangle showing ROI position
- **Crop to ROI View** - Optional zoomed view of ROI region only
- **Real-time Updates** - Auto-apply as sliders move

### Event Rate Controller (via I_ErcModule)
- **Rate Limiting** - Target event rate control (events/second)
- **Bandwidth Management** - Prevent USB saturation
- **Rate Display** - Current and target rate monitoring

### Anti-Flicker Filter (via I_AntiFlickerModule)
- **Filter Modes** - BAND_STOP / BAND_PASS
- **Frequency Band** - Configurable low/high frequency range
- **Duty Cycle** - Adjustable duty cycle percentage
- **Quick Presets** - 50Hz, 60Hz, 100Hz, 120Hz buttons

### Event Trail Filter (via I_EventTrailFilterModule)
- **Filter Types** - TRAIL / STC_CUT_TRAIL / STC_KEEP_TRAIL
- **Threshold Delay** - Adjustable time between bursts (μs)
- **Noise Reduction** - Filter trailing events from bursts

### Digital Crop (via I_DigitalCrop)
- **Crop Region** - Hardware-level sensor cropping
- **Position Control** - X/Y sliders for crop position
- **Size Control** - Width/Height sliders for crop size
- **Real-time Updates** - Auto-apply crop region changes

---

## Available But NOT Currently Exposed

### 1. Trigger In/Out - I_TriggerIn / I_TriggerOut
**Purpose:** Synchronize camera with external signals

**Features:**
- Configure external trigger inputs
- Configure trigger outputs
- Enable/disable triggers
- Set trigger channels

**Use Cases:**
- Multi-camera synchronization
- Sync with external devices
- Time-stamped captures
- Strobe lighting control

**Access:**
- `camera.get_device().get_facility<Metavision::I_TriggerIn>()`
- `camera.get_device().get_facility<Metavision::I_TriggerOut>()`

---

## GUI Implementation Status

### ✓ Implemented (High Priority)
1. **ROI (Region of Interest)** ✓ - Very useful for focusing on specific areas
2. **Event Rate Controller (ERC)** ✓ - Important for high-speed scenes
3. **Hardware Monitoring** ✓ - Temperature/illumination display

### ✓ Implemented (Medium Priority)
4. **Anti-Flicker Module** ✓ - Useful for indoor environments
5. **Event Trail Filter** ✓ - Good for noise reduction

### ✓ Implemented (Low Priority)
6. **Digital Crop** ✓ - Similar to ROI but less flexible

### ⏳ Not Yet Implemented
7. **Trigger In/Out** - Advanced use cases only (multi-camera sync)

---

## Implementation Notes

### Checking Facility Availability
Not all facilities are available on all cameras. Always check if facility exists:

```cpp
auto* roi = camera.get_device().get_facility<Metavision::I_ROI>();
if (roi) {
    // ROI is available, use it
    roi->enable(true);
} else {
    // ROI not supported on this camera
}
```

### Thread Safety
All facility operations should be called from the same thread or properly synchronized.

### Error Handling
Most facility methods return `bool` for success/failure. Some throw exceptions.

---

## Example: Adding ROI to GUI

```cpp
// Get ROI facility
auto* roi = camera.get_device().get_facility<Metavision::I_ROI>();
if (roi) {
    // In ImGui settings panel:
    if (ImGui::CollapsingHeader("Region of Interest")) {
        static bool roi_enabled = false;
        static int roi_x = 0, roi_y = 0;
        static int roi_width = 640, roi_height = 480;

        if (ImGui::Checkbox("Enable ROI", &roi_enabled)) {
            roi->enable(roi_enabled);
        }

        if (ImGui::SliderInt("X", &roi_x, 0, 1280)) {}
        if (ImGui::SliderInt("Y", &roi_y, 0, 720)) {}
        if (ImGui::SliderInt("Width", &roi_width, 1, 1280)) {}
        if (ImGui::SliderInt("Height", &roi_height, 1, 720)) {}

        if (ImGui::Button("Apply ROI")) {
            Metavision::I_ROI::Window window(roi_x, roi_y, roi_width, roi_height);
            roi->set_window(window);
        }
    }
}
```
