cmake_minimum_required(VERSION 3.26)
project(EventCameraViewer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use local dependencies (self-contained)
set(DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    "${DEPS_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends"
)

# OpenGL/GLFW/GLEW directories
set(GLEW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/glew-2.1.0")
set(GLFW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/glfw-3.3.8.bin.WIN64")

# Library directories
link_directories(
    "${DEPS_DIR}/lib"
    "${GLEW_DIR}/lib/Release/x64"
    "${GLFW_DIR}/lib-vc2022"
)

# Detect which libraries are available
if(EXISTS "${DEPS_DIR}/lib/opencv_world480.lib")
    set(USE_OPENCV_WORLD TRUE)
    message(STATUS "Using monolithic OpenCV (opencv_world480)")
else()
    set(USE_OPENCV_WORLD FALSE)
    message(STATUS "Using modular OpenCV libraries")
endif()

if(EXISTS "${DEPS_DIR}/lib/metavision_sdk_driver.lib")
    set(USE_SDK_DRIVER TRUE)
    message(STATUS "Using metavision_sdk_driver")
else()
    set(USE_SDK_DRIVER FALSE)
    message(STATUS "Using metavision_sdk_stream (legacy)")
endif()

if(EXISTS "${GLFW_DIR}/lib-vc2022/glfw3dll.lib")
    set(GLFW_LIB glfw3dll)
    message(STATUS "Using glfw3dll")
elseif(EXISTS "${GLFW_DIR}/lib-vc2022/glfw3.lib")
    set(GLFW_LIB glfw3)
    message(STATUS "Using glfw3 (static)")
endif()

# ImGui source files
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui")
set(IMGUI_SOURCES
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_demo.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
)

# Main executable - minimal event camera viewer
add_executable(event_camera_viewer
    src/main.cpp
    src/camera_manager.cpp
    src/app_config.cpp
    src/event_camera_genetic_optimizer.cpp
    # Core module (application state)
    src/core/frame_sync.cpp
    src/core/event_metrics.cpp
    src/core/display_settings.cpp
    src/core/camera_state.cpp
    src/core/app_state.cpp
    # Video processing module
    src/video/frame_buffer.cpp
    src/video/frame_processor.cpp
    src/video/texture_manager.cpp
    src/video/filters/roi_filter.cpp
    src/video/filters/subtraction_filter.cpp
    # Camera module (hardware features)
    src/camera/feature_manager.cpp
    src/camera/bias_manager.cpp
    src/camera/camera_controller.cpp
    src/camera/event_processor.cpp
    src/camera/features/monitoring_feature.cpp
    src/camera/features/erc_feature.cpp
    src/camera/features/antiflicker_feature.cpp
    src/camera/features/trail_filter_feature.cpp
    src/camera/features/roi_feature.cpp
    src/camera/features/digital_crop_feature.cpp
    # UI module (panels)
    src/ui/settings_panel.cpp
    src/ui/features_panel.cpp
    src/ui/camera_feed_panel.cpp
    src/ui/ga_panel.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(event_camera_viewer PRIVATE
    "${GLEW_DIR}/include"
    "${GLFW_DIR}/include"
)

# Build Metavision SDK library list
set(METAVISION_LIBS
    metavision_sdk_base
    metavision_sdk_core
    metavision_sdk_ui
    metavision_hal
    metavision_hal_discovery
)

if(USE_SDK_DRIVER)
    list(APPEND METAVISION_LIBS metavision_sdk_driver)
else()
    list(APPEND METAVISION_LIBS metavision_sdk_stream)
endif()

# Build OpenCV library list
if(USE_OPENCV_WORLD)
    set(OPENCV_LIBS opencv_world480)
else()
    set(OPENCV_LIBS
        opencv_core4
        opencv_imgproc4
        opencv_highgui4
        opencv_imgcodecs4
    )
endif()

target_link_libraries(event_camera_viewer
    ${METAVISION_LIBS}
    ${OPENCV_LIBS}
    ${GLFW_LIB}
    opengl32
    glew32
)

# Copy DLLs, plugins, and config file to output directory
if(WIN32)
    add_custom_command(TARGET event_camera_viewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${DEPS_DIR}/lib"
        $<TARGET_FILE_DIR:event_camera_viewer>
        COMMENT "Copying DLLs to output directory"
    )
    add_custom_command(TARGET event_camera_viewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/plugins"
        $<TARGET_FILE_DIR:event_camera_viewer>/plugins
        COMMENT "Copying Metavision plugins to output directory"
    )
    add_custom_command(TARGET event_camera_viewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/event_config.ini"
        $<TARGET_FILE_DIR:event_camera_viewer>/event_config.ini
        COMMENT "Copying config file to output directory"
    )
    # Copy GLFW DLL if using dynamic linking
    if(EXISTS "${GLFW_DIR}/lib-vc2022/glfw3.dll")
        add_custom_command(TARGET event_camera_viewer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLFW_DIR}/lib-vc2022/glfw3.dll"
            $<TARGET_FILE_DIR:event_camera_viewer>
            COMMENT "Copying GLFW DLL to output directory"
        )
    endif()
    # Copy GLEW DLL
    if(EXISTS "${GLEW_DIR}/bin/Release/x64/glew32.dll")
        add_custom_command(TARGET event_camera_viewer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLEW_DIR}/bin/Release/x64/glew32.dll"
            $<TARGET_FILE_DIR:event_camera_viewer>
            COMMENT "Copying GLEW DLL to output directory"
        )
    endif()
endif()

# Set output directory
set_target_properties(event_camera_viewer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
