# Event Camera Viewer

A minimal, self-contained C++ application for viewing event camera feeds with basic settings control.

## Features

- **USB Camera Connection**: Automatically detects and connects to event cameras
- **Live Feed Display**: Real-time visualization of event camera output
- **Camera Settings UI**: Interactive controls for camera biases and parameters
  - Event detection threshold (bias_diff)
  - Refractory period (bias_refr)
  - Photoreceptor follower (bias_fo)
  - High-pass filter (bias_hpf)
  - Pixel photoreceptor (bias_pr)
  - Frame accumulation time
- **Fully Self-Contained**: All dependencies included in repository

## Project Structure

```
EventCamera/
├── CMakeLists.txt           # Build configuration
├── tracking_config.ini      # Camera settings
├── include/
│   ├── camera_manager.h     # Camera connection interface
│   └── app_config.h         # Configuration management
├── src/
│   ├── main.cpp             # Main application
│   ├── camera_manager.cpp   # Camera initialization
│   └── app_config.cpp       # Config file parser
├── deps/                    # All runtime dependencies
│   ├── include/             # SDK headers (Metavision, OpenCV)
│   │   ├── metavision/      # Metavision SDK headers
│   │   └── opencv2/         # OpenCV headers
│   └── lib/                 # DLLs and libs (94 DLLs, 87 libs)
│       ├── *.dll            # Metavision, OpenCV, Boost
│       └── *.lib            # Import libraries
├── plugins/                 # Metavision camera plugins
│   └── silky_common_plugin.dll  # CenturyArks SilkyEvCam plugin
└── external/                # Third-party UI libraries
    ├── imgui/               # Dear ImGui UI framework
    ├── glfw-3.3.8.bin.WIN64/  # GLFW window library
    └── glew-2.1.0/          # OpenGL extension loader
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

### Basic Usage

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

Edit `tracking_config.ini` to set default camera parameters:

```ini
[Camera]
bias_diff = 128              # Event detection threshold (0-255)
bias_refr = 128              # Refractory period (0-255)
bias_fo = 128                # Photoreceptor follower (0-255)
bias_hpf = 128               # High-pass filter (0-255)
bias_pr = 128                # Pixel photoreceptor (0-255)
accumulation_time_s = 0.01   # Frame generation period (seconds)
```

## Usage

### Controls

- **Settings Panel** (left): Adjust camera biases and frame accumulation
  - Move sliders to change parameters
  - Click "Apply Bias Settings" to apply changes to camera
  - Click "Reset to Defaults" to restore default values

- **Camera Feed** (right): Live view from event camera
  - Automatically scales to fit window
  - Updates in real-time as events are captured

- **ESC or close window**: Exit application

### Camera Biases Explained

- **bias_diff**: Event detection sensitivity (higher = less sensitive)
- **bias_refr**: Time before pixel can trigger again
- **bias_fo**: Photoreceptor output buffer
- **bias_hpf**: Removes DC component from signal
- **bias_pr**: Photoreceptor amplification
- **accumulation_time_s**: How long to collect events before generating frame

## Hardware Support

- **CenturyArks SilkyEvCam HD**: Primary target hardware
- Any Metavision SDK compatible event camera should work

## Troubleshooting

### "No event cameras found"
- Check USB connection
- Verify Metavision SDK is properly installed
- Ensure camera drivers are loaded

### Build errors
- Verify `tracking` repository is in parent directory
- Check that all dependencies in `tracking/deps/` exist
- Ensure GLFW and GLEW are in `tracking/external/`

### Camera won't start
- Try different USB port
- Check camera permissions
- Verify no other application is using the camera

## License

This is a minimal viewer application based on the tracking repository code.
