# Available Event Camera Features

This document lists the camera features available in the Metavision SDK that could be exposed in the Event Camera Viewer GUI.

## Currently Exposed Features ✓

### Camera Biases (via I_LL_Biases)
- **bias_diff** - Event detection threshold (0-255)
- **bias_refr** - Refractory period (0-255)
- **bias_fo** - Photoreceptor follower (0-255)
- **bias_hpf** - High-pass filter (0-255)
- **bias_pr** - Pixel photoreceptor (0-255)

### Frame Generation
- **accumulation_time_s** - Event accumulation period (0.001-0.1s)

---

## Available But NOT Currently Exposed

### 1. Region of Interest (ROI) - I_ROI
**Purpose:** Define rectangular regions to process/ignore events

**Features:**
- Set ROI mode: ROI (keep events inside) or RONI (discard events inside)
- Define rectangular windows (x, y, width, height)
- Support for multiple windows
- Enable/disable ROI
- Line-based ROI (row/column binary maps)

**Use Cases:**
- Focus processing on specific areas
- Reduce data bandwidth
- Filter out noisy regions
- Track specific zones

**Access:** `camera.get_device().get_facility<Metavision::I_ROI>()`

---

### 2. Event Rate Controller (ERC) - I_ErcModule
**Purpose:** Limit maximum event rate to prevent bandwidth saturation

**Features:**
- Enable/disable ERC
- Set target event rate (events/second)
- Set event count per time period
- Get count period (reference time window)
- Get min/max supported rates

**Use Cases:**
- Prevent USB bandwidth overload
- Limit processing load
- Handle high-speed scenes
- Stabilize event flow

**Access:** `camera.get_device().get_facility<Metavision::I_ErcModule>()`

---

### 3. Event Trail Filter - I_EventTrailFilterModule
**Purpose:** Filter noise from event bursts

**Filter Types:**
- **TRAIL** - Keep only first event from burst, filter trailing events
- **STC_CUT_TRAIL** - Keep only second event after polarity change, filter rest
- **STC_KEEP_TRAIL** - Keep all trailing events after polarity change, filter first

**Features:**
- Enable/disable filter
- Set filter type
- Set threshold delay (microseconds between bursts)
- Get min/max threshold values

**Use Cases:**
- Reduce noise from rapid flickering
- Clean up trailing artifacts
- Improve signal-to-noise ratio
- Handle high-frequency patterns

**Access:** `camera.get_device().get_facility<Metavision::I_EventTrailFilterModule>()`

---

### 4. Anti-Flicker Module - I_AntiFlickerModule
**Purpose:** Filter events caused by artificial lighting (50/60Hz flicker)

**Features:**
- Enable/disable anti-flicker
- Set filtering mode: BAND_PASS or BAND_STOP
- Set frequency band (low_freq, high_freq in Hz)
- Set duty cycle (percentage)
- Get min/max supported frequencies

**Use Cases:**
- Remove 50Hz/60Hz artificial light flicker
- Clean indoor scenes with fluorescent lighting
- Filter specific frequency bands
- Improve data quality under AC lighting

**Access:** `camera.get_device().get_facility<Metavision::I_AntiFlickerModule>()`

---

### 5. Hardware Monitoring - I_Monitoring
**Purpose:** Monitor sensor health and environmental conditions

**Features:**
- Get temperature (°C)
- Get illumination (lux)
- Get pixel dead time (refractory period in microseconds)

**Use Cases:**
- Monitor sensor temperature for overheating
- Check lighting conditions
- Verify refractory period settings
- Debug hardware issues

**Access:** `camera.get_device().get_facility<Metavision::I_Monitoring>()`

---

### 6. Digital Crop - I_DigitalCrop
**Purpose:** Crop the sensor output to a smaller resolution

**Features:**
- Set cropping region
- Enable/disable crop
- Get cropped geometry

**Use Cases:**
- Reduce data volume
- Focus on specific sensor area
- Improve processing speed

**Access:** `camera.get_device().get_facility<Metavision::I_DigitalCrop>()`

---

### 7. Trigger In/Out - I_TriggerIn / I_TriggerOut
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

## Recommended Priority for GUI Implementation

### High Priority (Most Useful)
1. **ROI (Region of Interest)** - Very useful for focusing on specific areas
2. **Event Rate Controller (ERC)** - Important for high-speed scenes
3. **Hardware Monitoring** - Temperature/illumination display

### Medium Priority
4. **Anti-Flicker Module** - Useful for indoor environments
5. **Event Trail Filter** - Good for noise reduction

### Low Priority
6. **Digital Crop** - Similar to ROI but less flexible
7. **Trigger In/Out** - Advanced use cases only

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
