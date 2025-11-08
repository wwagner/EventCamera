# Event Camera Viewer

A comprehensive C++ application for viewing and controlling event camera feeds with advanced hardware feature access through the Metavision SDK.

## Features

### Core Functionality
- **USB Camera Connection**: Automatically detects and connects to event cameras
- **Live Feed Display**: Real-time visualization of event camera output with OpenGL rendering
- **Dynamic Bias Controls**: Hardware-aware bias sliders with camera-specific ranges
  - Event detection threshold (bias_diff)
  - Refractory period (bias_refr)
  - Photoreceptor follower (bias_fo)
  - High-pass filter (bias_hpf)
  - Pixel photoreceptor (bias_pr)
- **Frame Generation Control**: Adjustable event accumulation time (0.001-0.1 seconds)

### Advanced Camera Features

#### Hardware Monitoring
- **Temperature Monitoring**: Real-time sensor temperature display (°C)
- **Pixel Dead Time**: Refractory period monitoring (microseconds)
- Automatic capability detection for hardware-specific features

#### Region of Interest (ROI)
- Define rectangular regions to process/filter events
- **Real-Time Visual Feedback**: Green rectangle overlay showing ROI window
- **Crop to ROI View**: Option to display only the ROI region (zoomed view)
- Interactive sliders for precise positioning
- Live updates as sliders are adjusted
- Enable/disable ROI filtering on-the-fly

#### Event Rate Controller (ERC)
- Limit maximum event rate to prevent bandwidth saturation
- Configurable target event rate (events/second)
- Helps manage high-speed scenes and USB bandwidth

#### Anti-Flicker Filter
- Remove artificial lighting flicker (50Hz/60Hz AC lighting)
- **Filter Modes**:
  - BAND_STOP: Remove specific frequencies
  - BAND_PASS: Keep only specific frequencies
- **Quick Presets**: One-click buttons for common frequencies
  - 50Hz (Europe/Asia standard)
  - 60Hz (North America standard)
  - 100Hz and 120Hz harmonics
- Adjustable frequency band and duty cycle
- Real-time filter updates

#### Event Trail Filter
- Filter noise from event bursts and rapid flickering
- **Filter Types**:
  - TRAIL: Keep only first event from burst, filter trailing events
  - STC_CUT_TRAIL: Keep only second event after polarity change
  - STC_KEEP_TRAIL: Keep all trailing events after polarity change
- Adjustable threshold delay (microseconds between bursts)
- Real-time threshold updates
- Improves signal-to-noise ratio

#### Digital Crop
- Crop sensor output to reduce resolution and data volume
- Hardware-level cropping (all pixels outside dropped by sensor)
- Adjustable crop region (X, Y, Width, Height)
- Real-time crop region updates
- Reduces processing load and improves performance
- Similar to ROI but less flexible (entire sensor resolution reduced)

### Display Processing Features

#### Binary Stream Processing
- **Early 1-bit Conversion**: Ultra-fast LUT-based pixel range filtering
- **Stream Modes**:
  - OFF: Normal 8-bit passthrough (no filtering)
  - DOWN: Show only Range 3 pixels [96-127] (mid-range events)
  - UP: Show only Range 7 pixels [224-255] (bright events)
  - UP_DOWN: Show both ranges combined (mid + bright events)
- **Performance Optimized**:
  - Lookup table operations (~1-2 CPU cycles per pixel)
  - Single-channel output (1/3 memory bandwidth: 900KB vs 2.7MB)
  - Early conversion (happens once at source before all processing)
  - Cache-friendly (256-byte LUT fits in L1 cache)
- **GA Integration**: Works with Genetic Algorithm optimization when "Use Processed Pixels" is enabled
- **Grayscale Mode**: Convert output to single-channel grayscale for simpler visualization

**Use Cases**:
- Isolate specific event intensity ranges for analysis
- Reduce noise by filtering out unwanted pixel ranges
- Improve processing performance with 1-bit data representation
- Focus GA optimization on specific event types (mid-range vs bright)

### User Interface
- **ImGui-Based Controls**: Clean, responsive settings panel
- **Thread-Safe Updates**: Mutex-protected frame generation and camera control
- **Error Handling**: Graceful handling of unsupported features
- **Console Feedback**: Real-time status messages and debug information

## Project Structure

```
EventCamera/
├── CMakeLists.txt              # Build configuration
├── event_config.ini            # Camera settings (auto-copied to build)
├── run_event_camera.bat        # Quick launcher script
├── AVAILABLE_CAMERA_FEATURES.md # Documentation of all Metavision SDK features
├── include/
│   ├── camera_manager.h        # Camera connection interface
│   └── app_config.h            # Configuration management
├── src/
│   ├── main.cpp                # Main application with GUI
│   ├── camera_manager.cpp      # Camera initialization
│   └── app_config.cpp          # Config file parser
├── deps/                       # All runtime dependencies
│   ├── include/                # SDK headers (Metavision, OpenCV)
│   │   ├── metavision/         # Metavision SDK headers
│   │   └── opencv2/            # OpenCV headers
│   └── lib/                    # DLLs and libs (94 DLLs, 87 libs)
│       ├── *.dll               # Metavision, OpenCV, Boost
│       └── *.lib               # Import libraries
├── plugins/                    # Metavision camera plugins
│   └── silky_common_plugin.dll # CenturyArks SilkyEvCam plugin
└── external/                   # Third-party UI libraries
    ├── imgui/                  # Dear ImGui UI framework
    ├── glfw-3.3.8.bin.WIN64/   # GLFW window library
    └── glew-2.1.0/             # OpenGL extension loader
```

## Dependencies (All Included)

This project is **fully self-contained** with all dependencies included:
- **Metavision SDK**: Event camera drivers (Prophesee/CenturyArks)
  - Includes silky_common_plugin.dll for CenturyArks SilkyEvCam
- **OpenCV 4.8.0**: Image processing libraries
- **Boost 1.78**: C++ utility libraries
- **GLFW 3.3.8**: Window management
- **GLEW 2.1.0**: OpenGL extension loader
- **ImGui 1.90**: User interface framework

**Total Dependencies**: 94 DLLs, 87 import libraries, complete SDK headers

## Building

### Prerequisites

1. **CMake 3.26 or higher**
2. **Visual Studio 2022** (on Windows) with C++ development tools
3. **OpenGL support** (usually included with graphics drivers)

**Note**: No external SDK installation required - all dependencies are included!

### Build Steps

```bash
# From the EventCamera directory
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The executable will be in `build/bin/Release/event_camera_viewer.exe`

## Running

### Quick Start (Recommended)

Use the included launcher script from the repository root:

```bash
run_event_camera.bat
```

This automatically navigates to the correct directory and launches the application.

### Manual Launch

```bash
cd build/bin/Release
./event_camera_viewer.exe
```

The application will:
1. Scan for connected event cameras
2. Display available cameras and their serial numbers
3. Connect to the first available camera
4. Show live feed with settings panel

### Configuration

Edit `event_config.ini` in the repository root to set default camera parameters:

```ini
[Camera]
# Event camera bias settings (0-255)
bias_diff = 128       # Event detection threshold - higher = less sensitive
bias_refr = 128       # Refractory period - prevents rapid re-triggering
bias_fo = 128         # Photoreceptor follower - output buffer strength
bias_hpf = 128        # High-pass filter - removes DC component
bias_pr = 128         # Pixel photoreceptor - amplification gain

# Frame generation settings
accumulation_time_s = 0.01    # Event accumulation period in seconds (range: 0.001-0.1)
                              # Lower values = more responsive but noisier
                              # Higher values = smoother but more latency
```

**Note**: The config file is automatically copied to the build directory during compilation.

## Usage

### Camera Biases

- **Dynamic Ranges**: Bias sliders automatically adjust to camera-specific hardware limits
- **Real-Time Updates**: Changes can be applied on-the-fly without restarting
- **Reset Button**: Restore to middle of valid range

**Bias Explanations**:
- **bias_diff**: Event detection sensitivity (higher = less sensitive to changes)
- **bias_refr**: Time before pixel can trigger again (refractory period)
- **bias_fo**: Photoreceptor output buffer strength
- **bias_hpf**: High-pass filter - removes DC component from signal
- **bias_pr**: Photoreceptor amplification gain

### Frame Generation

- **Accumulation Time**: Controls how long events are collected before generating a frame
  - Lower values (0.001s): Very responsive, shows rapid changes, more noise
  - Higher values (0.1s): Smoother output, better noise reduction, more latency
- **Thread-Safe**: Updates are properly synchronized to prevent crashes

### Hardware Monitoring

View real-time sensor statistics:
- **Temperature**: Monitor sensor temperature (°C) to prevent overheating
- **Pixel Dead Time**: View current refractory period in microseconds
- Features only appear if supported by your camera

### Region of Interest (ROI)

Control which sensor regions generate events:

1. **Enable ROI**: Check the "Enable ROI" box
2. **Position Controls**: Use X/Y sliders to move the ROI window
3. **Size Controls**: Use Width/Height sliders to resize the region
4. **Visual Feedback**: Green rectangle overlay shows current ROI position
5. **Crop to ROI**: Enable "Crop View to ROI" to zoom into the selected region

**ROI Updates**: The ROI window moves in real-time as you drag sliders (when ROI is enabled)

**Use Cases**:
- Focus processing on specific areas to reduce CPU load
- Reduce data bandwidth for high-speed recording
- Filter out noisy regions (e.g., background areas)
- Track specific zones of interest

### Event Rate Controller (ERC)

Manage event throughput:

1. **Enable ERC**: Check the "Enable ERC" box
2. **Set Event Rate**: Adjust the target events/second slider
3. Monitor the current event rate in the status display

**Use Cases**:
- Prevent USB bandwidth overload in high-speed scenes
- Limit processing load for real-time applications
- Stabilize event flow for consistent performance

### Anti-Flicker Filter

Remove artificial lighting artifacts:

1. **Enable Anti-Flicker**: Check the "Enable Anti-Flicker" box
2. **Select Mode**:
   - **BAND_STOP**: Remove events at specific frequencies (most common)
   - **BAND_PASS**: Keep only events at specific frequencies
3. **Quick Presets**: Click preset buttons for common frequencies
   - **50Hz**: European/Asian AC lighting standard
   - **60Hz**: North American AC lighting standard
   - **100Hz/120Hz**: Second harmonics
4. **Manual Control**: Adjust frequency band (low/high) and duty cycle sliders

**Use Cases**:
- Clean up indoor scenes with fluorescent or LED lighting
- Remove 50Hz/60Hz flicker from videos
- Improve signal quality in office/industrial environments

### Event Trail Filter

Reduce noise from event bursts:

1. **Enable Trail Filter**: Check the "Enable Trail Filter" box
2. **Select Filter Type**:
   - **TRAIL**: Keep only first event from burst, filter trailing events (most common)
   - **STC_CUT_TRAIL**: Keep only second event after polarity change, filter rest
   - **STC_KEEP_TRAIL**: Keep all trailing events after polarity change, filter first
3. **Threshold Delay**: Adjust the threshold slider (microseconds)
   - Lower threshold = more aggressive filtering
   - Higher threshold = keep more events

**Use Cases**:
- Reduce noise from rapid flickering
- Clean up trailing artifacts from moving objects
- Improve signal-to-noise ratio in high-speed scenes
- Handle high-frequency patterns and vibrations

### Digital Crop

Reduce sensor resolution to improve performance:

1. **Enable Digital Crop**: Check the "Enable Digital Crop" box
2. **Position Controls**: Use X/Y sliders to position the crop region
3. **Size Controls**: Use Width/Height sliders to set crop region size
4. **Real-Time Updates**: Crop region applies as you adjust sliders (when enabled)

**Important Notes**:
- Digital crop reduces the entire sensor resolution (hardware-level)
- All pixels outside crop region are dropped by the sensor
- Similar to ROI but less flexible (cannot have multiple regions)
- Best used when you need to permanently reduce resolution for performance

**Use Cases**:
- Reduce data volume for recording or transmission
- Focus on specific sensor area permanently
- Improve processing speed for real-time applications
- Reduce USB bandwidth usage

### Binary Stream Processing

Control pixel range filtering for advanced event analysis:

1. **Select Stream Mode**: Use the "Binary Stream Mode" dropdown
   - **OFF**: No filtering (default 8-bit display)
   - **DOWN**: Show only mid-range pixels [96-127]
   - **UP**: Show only bright pixels [224-255]
   - **UP_DOWN**: Show both mid-range and bright pixels
2. **Real-Time Display**: Processing applies immediately to live view
3. **GA Integration**: Enable "Use Processed Pixels" in GA settings to apply filtering during optimization

**Technical Details**:
- Uses ultra-fast lookup tables for O(1) pixel conversion
- Reduces memory bandwidth by 66% (single-channel vs BGR)
- Early conversion happens before all other processing
- Maintains full compatibility with grayscale mode and GA optimization

**Use Cases**:
- Isolate specific event types for targeted analysis
- Reduce computational load with 1-bit data representation
- Filter noise by excluding unwanted intensity ranges
- Optimize GA fitness evaluation on specific pixel ranges

### Keyboard Controls

- **ESC**: Exit application
- **Close window**: Exit application

## Hardware Support

- **CenturyArks SilkyEvCam HD**: Primary target hardware
- **Prophesee Event Cameras**: Any Metavision SDK compatible camera
- Feature availability depends on camera model (automatically detected)

## Troubleshooting

### "No event cameras found"
- Check USB connection (USB 3.0 recommended)
- Try unplugging camera, waiting 5 seconds, then replugging
- Verify no other application is using the camera
- Check Device Manager for USB errors

### Application crashes on startup
- Ensure previous instances are closed: `taskkill /F /IM event_camera_viewer.exe`
- Try unplugging and replugging the camera
- Check that event_config.ini exists in the build directory
- Review console output for specific error messages

### "LIBUSB_ERROR_TIMEOUT"
- Close all instances of the application
- Unplug camera, wait 5 seconds, replug
- Try a different USB 3.0 port
- Restart computer if problem persists

### "Failed to get illumination" errors
- This is normal - many cameras don't have illumination sensors
- The application automatically detects and disables unsupported features
- These warnings are harmless and can be ignored

### Settings don't apply / crashes when clicking buttons
- Ensure camera is properly connected before adjusting settings
- Preset buttons automatically clamp to camera's supported range
- Check console for error messages indicating unsupported features

### ROI visualization not updating
- Ensure "Enable ROI" is checked
- ROI only updates in real-time when enabled
- Try disabling and re-enabling ROI if visualization seems stuck

## Advanced Features Documentation

For a complete list of available Metavision SDK camera features (including those not yet implemented in the GUI), see:
- **AVAILABLE_CAMERA_FEATURES.md**: Comprehensive documentation of all facilities

Additional features available for future implementation:
- Trigger In/Out (multi-camera synchronization with external signals)
- Advanced ROI modes (multiple windows, line-based ROI)

## Technical Notes

### Thread Safety
- Frame generation is protected by mutex locks
- Camera settings can be adjusted from GUI thread safely
- Event callbacks run on separate thread from rendering

### Error Handling
- All facility access checks for hardware support
- Exceptions caught and logged to console
- Graceful degradation when features unavailable

### Performance
- OpenGL-accelerated rendering
- Efficient event-to-frame conversion
- Minimal CPU overhead for GUI updates

## License

This is a minimal viewer application for event cameras using the Metavision SDK.

## Contributing

When adding new features:
1. Check camera facility availability before use
2. Add try-catch blocks for exception handling
3. Update AVAILABLE_CAMERA_FEATURES.md if exposing new SDK features
4. Test with multiple camera models if possible
