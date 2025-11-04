@echo off
REM Event Camera Viewer Launcher
REM
REM This script launches the Event Camera Viewer application.
REM The application will automatically detect connected event cameras
REM and display a live feed with interactive settings controls.

echo.
echo ========================================
echo   Event Camera Viewer
echo ========================================
echo.

REM Change to the build output directory where the executable and DLLs are located
cd /d "%~dp0build\bin\Release"

REM Check if the executable exists
if not exist "event_camera_viewer.exe" (
    echo ERROR: event_camera_viewer.exe not found!
    echo.
    echo Please build the project first using:
    echo   cmake -B build -DCMAKE_BUILD_TYPE=Release
    echo   cmake --build build --config Release
    echo.
    pause
    exit /b 1
)

echo Starting Event Camera Viewer...
echo The application window will open shortly.
echo.
echo Camera Controls:
echo   - Use the Settings panel to adjust camera biases
echo   - Press ESC or close the window to exit
echo.

REM Run the application (start it and let batch file exit)
start "" "event_camera_viewer.exe"

REM Return to the original directory
cd /d "%~dp0"
