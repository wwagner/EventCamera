# Dependency Inventory

This document lists all dependencies included in the EventCamera repository.

## Summary

- **94 DLL files** - Runtime libraries
- **87 LIB files** - Import libraries
- **1 Camera plugin** - CenturyArks SilkyEvCam support
- **Complete SDK headers** - Metavision SDK and OpenCV
- **3 External libraries** - ImGui, GLFW, GLEW

## Directory Structure

### deps/include/ - SDK Headers

Contains complete header files for:
- **metavision/** - Metavision SDK headers (Prophesee event camera SDK)
  - Core algorithms, driver interface, HAL facilities
  - Event processing, frame generation, camera discovery
- **opencv2/** - OpenCV 4.8.0 headers
  - Core, imgproc, highgui, calib3d modules

### deps/lib/ - Runtime Libraries (94 DLLs, 87 LIBs)

#### Metavision SDK Libraries
Event camera driver and processing libraries:
- metavision_sdk_base.dll/lib
- metavision_sdk_core.dll/lib
- metavision_sdk_driver.dll/lib
- metavision_sdk_ui.dll/lib
- metavision_hal.dll/lib
- metavision_hal_discovery.dll/lib
- hal_plugin_prophesee.lib
- hal_evk4_sample_plugin.lib
- silky_common_plugin.dll (in deps/lib and plugins/)

#### OpenCV Libraries (4.8.0)
Computer vision and image processing:
- opencv_world480.dll/lib (monolithic build)
- opencv_core4.dll/lib
- opencv_imgproc4.dll/lib
- opencv_highgui4.dll/lib
- opencv_imgcodecs4.dll/lib
- opencv_videoio4.dll/lib
- opencv_calib3d4.dll/lib
- opencv_features2d4.dll/lib

#### Boost Libraries (1.78)
C++ utility libraries:
- boost_chrono-vc143-mt-x64-1_78.dll
- boost_date_time-vc143-mt-x64-1_78.dll
- boost_filesystem-vc143-mt-x64-1_78.dll
- boost_program_options-vc143-mt-x64-1_78.dll
- boost_regex-vc143-mt-x64-1_78.dll
- boost_system-vc143-mt-x64-1_78.dll
- boost_thread-vc143-mt-x64-1_78.dll
- boost_timer-vc143-mt-x64-1_78.dll
- (Plus corresponding .lib files)

#### Additional Dependencies
Supporting libraries for video, compression, and codecs:
- brotlicommon.dll/lib
- brotlidec.dll/lib
- deflate.dll/lib
- freetype.dll/lib
- harfbuzz.dll/lib
- hdf5.dll/lib
- jpeg62.dll/lib
- jxl.dll/lib
- jxl_threads.dll/lib
- leptonica-1.84.1.dll/lib
- libpng16.dll/lib
- libsharpyuv.dll/lib
- libtiff.dll/lib
- libwebp.dll/lib
- openjp2.dll/lib
- tesseract54.dll/lib
- tbb12.dll/lib
- zlib1.dll/lib
- And many more supporting libraries

### plugins/ - Camera Hardware Plugins

- **silky_common_plugin.dll** - CenturyArks SilkyEvCam HD camera plugin
  - Enables connection to CenturyArks event cameras
  - Required for camera detection and initialization
  - Loaded automatically by Metavision HAL at runtime

### external/ - Third-Party UI and Graphics Libraries

#### ImGui (Dear ImGui 1.90)
Immediate-mode GUI library:
- imgui/imgui.cpp
- imgui/imgui.h
- imgui/imgui_demo.cpp
- imgui/imgui_draw.cpp
- imgui/imgui_tables.cpp
- imgui/imgui_widgets.cpp
- imgui/backends/imgui_impl_glfw.cpp
- imgui/backends/imgui_impl_opengl3.cpp
- Additional headers and supporting files

#### GLFW (3.3.8)
Window management and input handling:
- glfw-3.3.8.bin.WIN64/include/ - Headers
- glfw-3.3.8.bin.WIN64/lib-vc2022/ - Libraries
  - glfw3.lib or glfw3dll.lib
  - glfw3.dll (if dynamic linking)

#### GLEW (2.1.0)
OpenGL extension loading:
- glew-2.1.0/include/ - Headers (GL/glew.h)
- glew-2.1.0/lib/Release/x64/ - glew32.lib
- glew-2.1.0/bin/Release/x64/ - glew32.dll

## Runtime Requirements

At runtime, the application needs:

1. **All DLLs from deps/lib/** (automatically copied to build/bin/Release/)
2. **plugins/silky_common_plugin.dll** (copied to build/bin/Release/plugins/)
3. **glew32.dll** (copied from external/glew-2.1.0/bin/)
4. **glfw3.dll** if using dynamic linking (copied from external/glfw/)
5. **tracking_config.ini** (camera configuration)

CMake automatically copies all required files during the build process.

## Platform Support

- **Windows x64**: Fully supported (all libraries included)
- **Linux/macOS**: Would require recompiling libraries for target platform

## License Notes

This repository includes third-party libraries with their own licenses:
- **Metavision SDK**: Prophesee/CenturyArks licensing
- **OpenCV**: Apache 2.0 License
- **Boost**: Boost Software License
- **ImGui**: MIT License
- **GLFW**: zlib/libpng License
- **GLEW**: Modified BSD License, MIT License

Please review individual library licenses before commercial use.

## Size Information

- Total repository size: ~500+ MB (with all dependencies)
- deps/lib/: ~450 MB (DLLs and libraries)
- external/: ~50 MB (GLFW, GLEW, ImGui)
- plugins/: ~3.5 MB (silky_common_plugin.dll)
- Headers: ~5 MB

## Updating Dependencies

To update to newer versions of libraries:

1. Replace DLLs and libs in `deps/lib/`
2. Update headers in `deps/include/`
3. Update version numbers in CMakeLists.txt and README.md
4. Test thoroughly with your event camera hardware

## Verification

To verify all dependencies are present:

```bash
# Check DLL count
ls deps/lib/*.dll | wc -l    # Should be 94
ls deps/lib/*.lib | wc -l    # Should be 87

# Check plugin
ls plugins/silky_common_plugin.dll  # Should exist

# Check external libraries
ls external/imgui/imgui.cpp         # Should exist
ls external/glfw-3.3.8.bin.WIN64/   # Should exist
ls external/glew-2.1.0/             # Should exist

# Check headers
ls deps/include/metavision/          # Should exist
ls deps/include/opencv2/             # Should exist
```

## Build Output

After building, `build/bin/Release/` will contain:
- event_camera_viewer.exe
- All 94 DLL files (from deps/lib/)
- plugins/silky_common_plugin.dll
- glew32.dll
- glfw3.dll (if dynamic)
- tracking_config.ini

Total output directory size: ~450 MB

This allows the application to run standalone without requiring any external SDK installation.
