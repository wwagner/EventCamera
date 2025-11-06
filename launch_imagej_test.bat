@echo off
REM ImageJ + Event Camera Test Launcher
REM
REM This script launches both Fiji/ImageJ and the Event Camera Viewer
REM for testing real-time streaming integration.

echo.
echo ========================================
echo   ImageJ + Event Camera Test Launcher
echo ========================================
echo.

REM Check if Event Camera executable exists
if not exist "build\bin\Release\event_camera_viewer.exe" (
    echo ERROR: event_camera_viewer.exe not found!
    echo Please build the project first using run_event_camera.bat
    echo.
    pause
    exit /b 1
)

REM Check if Fiji is installed
if not exist "C:\Program Files\Fiji\fiji-windows-x64.exe" (
    echo ERROR: Fiji not found at C:\Program Files\Fiji
    echo Please install Fiji or update the path in this script.
    echo.
    pause
    exit /b 1
)

echo Starting Fiji/ImageJ...
echo.
echo INSTRUCTIONS:
echo   1. In Fiji: Go to Plugins ^> Macros ^> Edit...
echo   2. Click File ^> Open and select: imagej_stream_monitor.ijm
echo   3. Click the "Run" button at the bottom of the macro editor
echo   4. Then switch to the Event Camera window
echo.
echo Starting Fiji now...

REM Start Fiji
start "Fiji ImageJ" "C:\Program Files\Fiji\fiji-windows-x64.exe"

REM Wait a moment for Fiji to start
timeout /t 3 /nobreak >nul

echo.
echo Starting Event Camera Viewer...
echo.
echo Camera Controls:
echo   - Streaming is ENABLED (10 FPS to ImageJ)
echo   - Stream directory: C:\Users\wolfw\OneDrive\Desktop\imagej_stream
echo   - Use Settings panel to adjust camera biases
echo   - Press ESC or close the window to exit
echo.

REM Change to the build directory and start Event Camera
cd /d "%~dp0build\bin\Release"
start "Event Camera Viewer" "event_camera_viewer.exe"

REM Return to original directory
cd /d "%~dp0"

echo.
echo Both applications launched!
echo.
echo Remember to load and run the macro in Fiji:
echo   imagej_stream_monitor.ijm
echo.
pause
