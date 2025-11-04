@echo off
echo ========================================
echo EventCamera Dependency Verification
echo ========================================
echo.

echo Checking directory structure...
if exist "deps\include\metavision" (echo [OK] Metavision headers) else (echo [FAIL] Metavision headers missing)
if exist "deps\include\opencv2" (echo [OK] OpenCV headers) else (echo [FAIL] OpenCV headers missing)
if exist "deps\lib" (echo [OK] deps/lib directory) else (echo [FAIL] deps/lib missing)
if exist "plugins\silky_common_plugin.dll" (echo [OK] SilkyEvCam plugin) else (echo [FAIL] Plugin missing)
if exist "external\imgui" (echo [OK] ImGui) else (echo [FAIL] ImGui missing)
if exist "external\glfw-3.3.8.bin.WIN64" (echo [OK] GLFW) else (echo [FAIL] GLFW missing)
if exist "external\glew-2.1.0" (echo [OK] GLEW) else (echo [FAIL] GLEW missing)

echo.
echo Checking source files...
if exist "include\camera_manager.h" (echo [OK] camera_manager.h) else (echo [FAIL] camera_manager.h missing)
if exist "include\app_config.h" (echo [OK] app_config.h) else (echo [FAIL] app_config.h missing)
if exist "src\main.cpp" (echo [OK] main.cpp) else (echo [FAIL] main.cpp missing)
if exist "src\camera_manager.cpp" (echo [OK] camera_manager.cpp) else (echo [FAIL] camera_manager.cpp missing)
if exist "src\app_config.cpp" (echo [OK] app_config.cpp) else (echo [FAIL] app_config.cpp missing)

echo.
echo Checking configuration...
if exist "tracking_config.ini" (echo [OK] tracking_config.ini) else (echo [FAIL] Config missing)
if exist "CMakeLists.txt" (echo [OK] CMakeLists.txt) else (echo [FAIL] CMakeLists.txt missing)

echo.
echo ========================================
echo Verification complete!
echo ========================================
echo.
echo To build the project:
echo   cmake -B build -DCMAKE_BUILD_TYPE=Release
echo   cmake --build build --config Release
echo.
pause
