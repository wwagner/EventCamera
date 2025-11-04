/**
 * Minimal Event Camera Viewer
 *
 * Simple application to view event camera feeds with basic settings control.
 * Connects to event cameras via USB and displays live feed with ImGui controls.
 */

#include <iostream>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>

// OpenGL/GLFW/ImGui
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// Metavision SDK
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/hal/facilities/i_ll_biases.h>

// Local headers
#include "camera_manager.h"
#include "app_config.h"

// Global state
std::atomic<bool> running{true};
cv::Mat current_frame;
std::mutex frame_mutex;
GLuint texture_id = 0;
int image_width = 1280;
int image_height = 720;

/**
 * Create OpenGL texture from OpenCV Mat
 */
void update_texture(const cv::Mat& frame) {
    if (frame.empty()) return;

    std::lock_guard<std::mutex> lock(frame_mutex);
    current_frame = frame.clone();
}

/**
 * Upload OpenCV frame to OpenGL texture
 */
void upload_frame_to_gpu() {
    std::lock_guard<std::mutex> lock(frame_mutex);

    if (current_frame.empty()) return;

    // Ensure texture is created
    if (texture_id == 0) {
        glGenTextures(1, &texture_id);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
        glBindTexture(GL_TEXTURE_2D, texture_id);
    }

    // Convert BGR to RGB if needed
    cv::Mat rgb_frame;
    if (current_frame.channels() == 3) {
        cv::cvtColor(current_frame, rgb_frame, cv::COLOR_BGR2RGB);
    } else {
        rgb_frame = current_frame;
    }

    // Upload to GPU
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb_frame.cols, rgb_frame.rows,
                 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_frame.data);
}

/**
 * Apply camera bias settings
 */
void apply_bias_settings(Metavision::Camera& camera, const AppConfig::CameraSettings& settings) {
    auto* i_ll_biases = camera.get_facility<Metavision::I_LL_Biases>();
    if (i_ll_biases) {
        i_ll_biases->set("bias_diff", settings.bias_diff);
        i_ll_biases->set("bias_refr", settings.bias_refr);
        i_ll_biases->set("bias_fo", settings.bias_fo);
        i_ll_biases->set("bias_hpf", settings.bias_hpf);
        i_ll_biases->set("bias_pr", settings.bias_pr);
        std::cout << "Applied camera biases: diff=" << settings.bias_diff
                  << " refr=" << settings.bias_refr
                  << " fo=" << settings.bias_fo
                  << " hpf=" << settings.bias_hpf
                  << " pr=" << settings.bias_pr << std::endl;
    }
}

/**
 * Main application
 */
int main(int argc, char* argv[]) {
    std::cout << "Event Camera Viewer v1.0" << std::endl;
    std::cout << "=========================" << std::endl;

    // Load configuration
    AppConfig config;
    if (!config.load("tracking_config.ini")) {
        std::cerr << "Warning: Could not load config file, using defaults" << std::endl;
    }

    // List available cameras
    std::cout << "\nScanning for event cameras..." << std::endl;
    auto available_cameras = CameraManager::list_available_cameras();

    if (available_cameras.empty()) {
        std::cerr << "ERROR: No event cameras found!" << std::endl;
        std::cerr << "Please ensure camera is connected via USB." << std::endl;
        return -1;
    }

    std::cout << "Found " << available_cameras.size() << " camera(s):" << std::endl;
    for (size_t i = 0; i < available_cameras.size(); ++i) {
        std::cout << "  [" << i << "] " << available_cameras[i] << std::endl;
    }

    // Initialize camera manager
    CameraManager camera_mgr;
    std::string serial1 = available_cameras[0];  // Use first camera
    int num_cameras = camera_mgr.initialize(serial1);

    if (num_cameras == 0) {
        std::cerr << "ERROR: Failed to initialize camera" << std::endl;
        return -1;
    }

    std::cout << "\nInitialized " << num_cameras << " camera(s)" << std::endl;

    // Get camera info
    auto& cam_info = camera_mgr.get_camera(0);
    image_width = cam_info.width;
    image_height = cam_info.height;

    std::cout << "Camera: " << cam_info.serial << std::endl;
    std::cout << "Resolution: " << image_width << "x" << image_height << std::endl;

    // Apply initial bias settings
    apply_bias_settings(*cam_info.camera, config.camera_settings());

    // Create frame generation algorithm
    const uint32_t accumulation_time_us = static_cast<uint32_t>(
        config.camera_settings().accumulation_time_s * 1000000);

    auto frame_gen = std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(
        image_width, image_height, accumulation_time_us);

    std::cout << "Frame accumulation time: " << config.camera_settings().accumulation_time_s
              << "s (" << accumulation_time_us << " us)" << std::endl;

    // Set up frame callback
    frame_gen->set_output_callback([](const Metavision::timestamp, cv::Mat& frame) {
        if (!frame.empty()) {
            update_texture(frame);
        }
    });

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "ERROR: Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create window
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    int window_width = 1600;
    int window_height = 900;
    GLFWwindow* window = glfwCreateWindow(window_width, window_height,
                                          "Event Camera Viewer", nullptr, nullptr);
    if (!window) {
        std::cerr << "ERROR: Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "ERROR: Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    std::cout << "\nStarting camera..." << std::endl;
    cam_info.camera->start();
    std::cout << "Camera started successfully!" << std::endl;
    std::cout << "\nPress ESC or close window to exit\n" << std::endl;

    // Camera event processing thread
    std::thread event_thread([&]() {
        auto& camera = cam_info.camera;

        // Set up event callback
        camera->cd().add_callback([&](const Metavision::EventCD* begin,
                                     const Metavision::EventCD* end) {
            frame_gen->process_events(begin, end);
        });

        // Process events while running
        while (running && camera->is_running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    // Track if settings changed
    bool settings_changed = false;
    auto previous_settings = config.camera_settings();

    // Main render loop
    while (!glfwWindowShouldClose(window) && running) {
        glfwPollEvents();

        // Handle ESC key
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            running = false;
        }

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Settings panel
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 500), ImGuiCond_FirstUseEver);

        if (ImGui::Begin("Camera Settings")) {
            ImGui::Text("Camera: %s", cam_info.serial.c_str());
            ImGui::Text("Resolution: %dx%d", image_width, image_height);
            ImGui::Text("FPS: %.1f", io.Framerate);
            ImGui::Separator();

            ImGui::Text("Camera Biases (0-255)");
            ImGui::Text("Adjust these to tune event detection");
            ImGui::Spacing();

            auto& cam_settings = config.camera_settings();

            if (ImGui::SliderInt("bias_diff", &cam_settings.bias_diff, 0, 255)) {
                settings_changed = true;
            }
            ImGui::TextWrapped("Event detection threshold - higher = less sensitive");
            ImGui::Spacing();

            if (ImGui::SliderInt("bias_refr", &cam_settings.bias_refr, 0, 255)) {
                settings_changed = true;
            }
            ImGui::TextWrapped("Refractory period - prevents rapid re-triggering");
            ImGui::Spacing();

            if (ImGui::SliderInt("bias_fo", &cam_settings.bias_fo, 0, 255)) {
                settings_changed = true;
            }
            ImGui::TextWrapped("Photoreceptor follower");
            ImGui::Spacing();

            if (ImGui::SliderInt("bias_hpf", &cam_settings.bias_hpf, 0, 255)) {
                settings_changed = true;
            }
            ImGui::TextWrapped("High-pass filter - reduces DC component");
            ImGui::Spacing();

            if (ImGui::SliderInt("bias_pr", &cam_settings.bias_pr, 0, 255)) {
                settings_changed = true;
            }
            ImGui::TextWrapped("Pixel photoreceptor");
            ImGui::Spacing();

            ImGui::Separator();
            ImGui::Text("Frame Generation");

            if (ImGui::SliderFloat("Accumulation (s)", &cam_settings.accumulation_time_s,
                                  0.001f, 0.1f, "%.3f")) {
                // Update frame generation period
                const uint32_t new_time_us = static_cast<uint32_t>(
                    cam_settings.accumulation_time_s * 1000000);
                frame_gen = std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(
                    image_width, image_height, new_time_us);
                frame_gen->set_output_callback([](const Metavision::timestamp, cv::Mat& frame) {
                    if (!frame.empty()) {
                        update_texture(frame);
                    }
                });
            }
            ImGui::TextWrapped("Time to accumulate events into frame");

            // Apply button
            ImGui::Spacing();
            ImGui::Separator();
            if (settings_changed) {
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "Settings changed!");
                if (ImGui::Button("Apply Bias Settings", ImVec2(-1, 0))) {
                    apply_bias_settings(*cam_info.camera, cam_settings);
                    previous_settings = cam_settings;
                    settings_changed = false;
                }
            }

            // Reset button
            if (ImGui::Button("Reset to Defaults", ImVec2(-1, 0))) {
                cam_settings.bias_diff = 128;
                cam_settings.bias_refr = 128;
                cam_settings.bias_fo = 128;
                cam_settings.bias_hpf = 128;
                cam_settings.bias_pr = 128;
                cam_settings.accumulation_time_s = 0.01f;
                settings_changed = true;
            }
        }
        ImGui::End();

        // Camera view window
        ImGui::SetNextWindowPos(ImVec2(420, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(window_width - 430, window_height - 20),
                                ImGuiCond_FirstUseEver);

        if (ImGui::Begin("Camera Feed")) {
            upload_frame_to_gpu();

            if (texture_id != 0) {
                ImVec2 window_size = ImGui::GetContentRegionAvail();

                // Maintain aspect ratio
                float aspect = static_cast<float>(image_width) / image_height;
                float display_width = window_size.x;
                float display_height = display_width / aspect;

                if (display_height > window_size.y) {
                    display_height = window_size.y;
                    display_width = display_height * aspect;
                }

                ImGui::Image((void*)(intptr_t)texture_id,
                           ImVec2(display_width, display_height));
            } else {
                ImGui::Text("Waiting for camera frames...");
            }
        }
        ImGui::End();

        // Render
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    std::cout << "\nShutting down..." << std::endl;
    running = false;

    // Stop camera
    if (cam_info.camera && cam_info.camera->is_running()) {
        cam_info.camera->stop();
    }

    // Wait for event thread
    if (event_thread.joinable()) {
        event_thread.join();
    }

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Cleanup OpenGL
    if (texture_id != 0) {
        glDeleteTextures(1, &texture_id);
    }

    // Cleanup GLFW
    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "Shutdown complete" << std::endl;
    return 0;
}
