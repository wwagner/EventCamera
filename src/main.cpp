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
#include <metavision/hal/facilities/i_roi.h>
#include <metavision/hal/facilities/i_erc_module.h>
#include <metavision/hal/facilities/i_monitoring.h>
#include <metavision/hal/facilities/i_antiflicker_module.h>
#include <metavision/hal/facilities/i_event_trail_filter_module.h>
#include <metavision/hal/facilities/i_digital_crop.h>

// Local headers
#include "camera_manager.h"
#include "app_config.h"
#include "event_camera_genetic_optimizer.h"

// Global state
std::atomic<bool> running{true};
cv::Mat current_frame;
std::mutex frame_mutex;
std::mutex framegen_mutex;  // Protects camera_state.frame_gen from race conditions
GLuint texture_id = 0;
int image_width = 1280;
int image_height = 720;

// ROI visualization
struct ROIVisualization {
    bool show = false;
    bool crop_to_roi = false;  // Show only ROI region
    int x = 0, y = 0, width = 640, height = 360;
    std::mutex mutex;
} roi_viz;

// Genetic Optimizer state
struct GAState {
    std::unique_ptr<EventCameraGeneticOptimizer> optimizer;
    std::unique_ptr<std::thread> optimizer_thread;
    std::atomic<bool> running{false};
    std::atomic<int> current_generation{0};
    std::atomic<float> best_fitness{1e9f};
    EventCameraGeneticOptimizer::Genome best_genome;
    EventCameraGeneticOptimizer::FitnessResult best_result;
    std::mutex mutex;

    // Frames for fitness evaluation
    std::vector<cv::Mat> captured_frames;
    std::mutex frames_mutex;
} ga_state;
// Camera hotplug state
struct CameraState {
    std::unique_ptr<CameraManager> camera_mgr;
    std::unique_ptr<Metavision::PeriodicFrameGenerationAlgorithm> frame_gen;
    std::unique_ptr<std::thread> event_thread;
    std::atomic<bool> camera_connected{false};
    std::atomic<bool> simulation_mode{false};
    std::mutex connection_mutex;
} camera_state;

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

    // Create a copy to draw on (don't modify the original)
    cv::Mat display_frame;

    // Crop to ROI or show full frame
    {
        std::lock_guard<std::mutex> roi_lock(roi_viz.mutex);
        if (roi_viz.show && roi_viz.crop_to_roi) {
            // Crop to ROI region only
            // Ensure ROI is within bounds
            int x = std::max(0, std::min(roi_viz.x, current_frame.cols - 1));
            int y = std::max(0, std::min(roi_viz.y, current_frame.rows - 1));
            int w = std::min(roi_viz.width, current_frame.cols - x);
            int h = std::min(roi_viz.height, current_frame.rows - y);

            if (w > 0 && h > 0) {
                cv::Rect roi_rect(x, y, w, h);
                display_frame = current_frame(roi_rect).clone();
            } else {
                display_frame = current_frame.clone();
            }
        } else {
            display_frame = current_frame.clone();

            // Draw ROI rectangle if enabled but not cropped
            if (roi_viz.show) {
                // Draw bright green rectangle
                cv::rectangle(display_frame,
                             cv::Point(roi_viz.x, roi_viz.y),
                             cv::Point(roi_viz.x + roi_viz.width, roi_viz.y + roi_viz.height),
                             cv::Scalar(0, 255, 0), 2);

                // Draw corner markers
                int marker_size = 10;
                cv::line(display_frame,
                        cv::Point(roi_viz.x, roi_viz.y),
                        cv::Point(roi_viz.x + marker_size, roi_viz.y),
                        cv::Scalar(0, 255, 0), 3);
                cv::line(display_frame,
                        cv::Point(roi_viz.x, roi_viz.y),
                        cv::Point(roi_viz.x, roi_viz.y + marker_size),
                        cv::Scalar(0, 255, 0), 3);
            }
        }
    }

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
    if (display_frame.channels() == 3) {
        cv::cvtColor(display_frame, rgb_frame, cv::COLOR_BGR2RGB);
    } else {
        rgb_frame = display_frame;
    }

    // Upload to GPU
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb_frame.cols, rgb_frame.rows,
                 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_frame.data);
}

/**
 * Apply camera bias settings
 */
void apply_bias_settings(Metavision::Camera& camera, const AppConfig::CameraSettings& settings) {
    auto* i_ll_biases = camera.get_device().get_facility<Metavision::I_LL_Biases>();
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

// Bias range structure  
struct BiasRange {
    int min, max, current;
};

/**
 * Attempt to connect to a camera (for hot-plug support)
 */

/**
 * Fitness evaluation callback for genetic algorithm
 * Applies genome parameters to camera and captures frames for evaluation
 */
EventCameraGeneticOptimizer::FitnessResult evaluate_genome_fitness(
    const EventCameraGeneticOptimizer::Genome& genome,
    AppConfig& config,
    int num_frames = 30) {
    
    EventCameraGeneticOptimizer::FitnessResult result;
    
    // Apply genome parameters to camera (if connected)
    if (camera_state.camera_connected && camera_state.camera_mgr) {
        auto& cam_info = camera_state.camera_mgr->get_camera(0);
        auto* i_ll_biases = cam_info.camera->get_device().get_facility<Metavision::I_LL_Biases>();
        
        if (i_ll_biases) {
            try {
                i_ll_biases->set("bias_diff", genome.bias_diff);
                i_ll_biases->set("bias_refr", genome.bias_refr);
                i_ll_biases->set("bias_fo", genome.bias_fo);
                i_ll_biases->set("bias_hpf", genome.bias_hpf);
                i_ll_biases->set("bias_pr", genome.bias_pr);
            } catch (const std::exception& e) {
                std::cerr << "Error applying biases: " << e.what() << std::endl;
                result.combined_fitness = 1e9f;  // Very bad fitness
                return result;
            }
        }
        
        // Update accumulation time
        if (camera_state.frame_gen) {
            const uint32_t accumulation_time_us = static_cast<uint32_t>(
                genome.accumulation_time_s * 1000000);
            
            std::lock_guard<std::mutex> lock(framegen_mutex);
            camera_state.frame_gen = std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(
                image_width, image_height, accumulation_time_us);
            camera_state.frame_gen->set_output_callback([](const Metavision::timestamp, cv::Mat& frame) {
                if (!frame.empty()) {
                    update_texture(frame);
                }
            });
        }
        
        // Apply other genome parameters (trail filter, antiflicker, ERC, etc.)
        if (genome.enable_trail_filter) {
            auto* trail_filter = cam_info.camera->get_device().get_facility<Metavision::I_EventTrailFilterModule>();
            if (trail_filter) {
                trail_filter->enable(true);
                trail_filter->set_threshold(genome.trail_threshold_us);
            }
        }
        
        if (genome.enable_antiflicker) {
            auto* antiflicker = cam_info.camera->get_device().get_facility<Metavision::I_AntiFlickerModule>();
            if (antiflicker) {
                antiflicker->enable(true);
                antiflicker->set_frequency_band(genome.af_low_freq, genome.af_high_freq);
            }
        }
        
        if (genome.enable_erc) {
            auto* erc = cam_info.camera->get_device().get_facility<Metavision::I_ErcModule>();
            if (erc) {
                erc->enable(true);
                erc->set_cd_event_rate(genome.erc_target_rate);
            }
        }
    }
    
    // Wait for parameters to stabilize
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Capture frames
    std::vector<cv::Mat> captured_frames;
    captured_frames.reserve(num_frames);
    
    for (int i = 0; i < num_frames; ++i) {
        cv::Mat captured_frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (!current_frame.empty()) {
                captured_frame = current_frame.clone();
            }
        }
        
        if (!captured_frame.empty()) {
            captured_frames.push_back(captured_frame);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(33));  // ~30 FPS
    }
    
    // Calculate fitness metrics from captured frames
    if (captured_frames.empty()) {
        result.combined_fitness = 1e9f;  // Very bad fitness
        return result;
    }
    
    float total_contrast = 0.0f;
    float total_noise = 0.0f;
    
    for (const auto& frame : captured_frames) {
        total_contrast += EventCameraGeneticOptimizer::calculate_contrast(frame);
        total_noise += EventCameraGeneticOptimizer::calculate_noise(frame);
    }
    
    result.contrast_score = total_contrast / captured_frames.size();
    result.noise_metric = total_noise / captured_frames.size();
    result.num_valid_frames = captured_frames.size();
    
    // Combined fitness: minimize (1/contrast + noise)
    // We want high contrast and low noise
    const float alpha = 0.5f;  // Weight for contrast
    const float beta = 0.5f;   // Weight for noise
    result.combined_fitness = alpha * (1.0f / (result.contrast_score + 1.0f)) + beta * result.noise_metric;
    
    return result;
}


/**
 * Attempt to connect to a camera (for hot-plug support)
 */
bool try_connect_camera(AppConfig& config, std::map<std::string, BiasRange>& bias_ranges,
                        const std::string& serial_hint = "") {
    std::lock_guard<std::mutex> lock(camera_state.connection_mutex);
    
    if (camera_state.camera_connected) {
        std::cout << "Camera already connected" << std::endl;
        return true;
    }
    
    std::cout << "Scanning for cameras..." << std::endl;
    auto available_cameras = CameraManager::list_available_cameras();
    
    if (available_cameras.empty()) {
        std::cout << "No cameras found" << std::endl;
        return false;
    }
    
    std::cout << "Found camera, attempting to connect..." << std::endl;
    
    // Stop simulation thread if running
    if (camera_state.simulation_mode && camera_state.event_thread) {
        running = false;
        if (camera_state.event_thread->joinable()) {
            camera_state.event_thread->join();
        }
        running = true;
    }
    
    // Initialize camera manager
    camera_state.camera_mgr = std::make_unique<CameraManager>();
    int num_cameras = camera_state.camera_mgr->initialize(serial_hint);
    
    if (num_cameras == 0) {
        std::cerr << "Failed to initialize camera" << std::endl;
        camera_state.camera_mgr = nullptr;
        return false;
    }
    
    auto& cam_info = camera_state.camera_mgr->get_camera(0);
    image_width = cam_info.width;
    image_height = cam_info.height;
    
    // Query bias ranges from newly connected camera
    bias_ranges.clear();
    auto* i_ll_biases = cam_info.camera->get_device().get_facility<Metavision::I_LL_Biases>();
    if (i_ll_biases) {
        for (const auto& name : {"bias_diff", "bias_refr", "bias_fo", "bias_hpf", "bias_pr"}) {
            try {
                Metavision::LL_Bias_Info info;
                if (i_ll_biases->get_bias_info(name, info)) {
                    auto range = info.get_bias_range();
                    int current = i_ll_biases->get(name);
                    bias_ranges[name] = {range.first, range.second, current};
                }
            } catch (...) {
                // Bias not available on this camera
            }
        }
    }
    
    // Create frame generator
    const uint32_t accumulation_time_us = static_cast<uint32_t>(
        config.camera_settings().accumulation_time_s * 1000000);
    
    camera_state.frame_gen = std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(
        image_width, image_height, accumulation_time_us);
    
    camera_state.frame_gen->set_output_callback([](const Metavision::timestamp, cv::Mat& frame) {
        if (!frame.empty()) {
            update_texture(frame);
        }
    });
    
    // Start camera
    cam_info.camera->start();
    std::cout << "Camera started: " << cam_info.serial << std::endl;
    
    // Start camera event thread
    camera_state.event_thread = std::make_unique<std::thread>([&]() {
        auto& cam_info = camera_state.camera_mgr->get_camera(0);
        auto& camera = cam_info.camera;
        
        camera->cd().add_callback([&](const Metavision::EventCD* begin,
                                     const Metavision::EventCD* end) {
            std::lock_guard<std::mutex> lock(framegen_mutex);
            if (camera_state.frame_gen) {
                camera_state.frame_gen->process_events(begin, end);
            }
        });
        
        while (running && camera->is_running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
    
    camera_state.camera_connected = true;
    camera_state.simulation_mode = false;
    
    return true;
}

/**
 * Main application
 */
int main(int argc, char* argv[]) {
    std::cout << "Event Camera Viewer v1.0" << std::endl;
    std::cout << "=========================" << std::endl;

    // Load configuration
    AppConfig config;
    if (!config.load("event_config.ini")) {
        std::cerr << "Warning: Could not load config file, using defaults" << std::endl;
    }

    // Camera/Simulation Mode

    // List available cameras
    std::cout << "\nScanning for event cameras..." << std::endl;
    auto available_cameras = CameraManager::list_available_cameras();

    if (available_cameras.empty()) {
        std::cout << "No event cameras found - starting in SIMULATION MODE" << std::endl;
        std::cout << "You can connect a camera later using the 'Connect Camera' button" << std::endl;
        camera_state.simulation_mode = true;
        image_width = 1280;
        image_height = 720;
    } else {
        std::cout << "Found " << available_cameras.size() << " camera(s):" << std::endl;
        for (size_t i = 0; i < available_cameras.size(); ++i) {
            std::cout << "  [" << i << "] " << available_cameras[i] << std::endl;
        }

        // Initialize camera manager
        camera_state.camera_mgr = std::make_unique<CameraManager>();
        std::string serial1 = available_cameras[0];  // Use first camera
        int num_cameras = camera_state.camera_mgr->initialize(serial1);

        if (num_cameras == 0) {
            std::cerr << "ERROR: Failed to initialize camera - starting in SIMULATION MODE" << std::endl;
            camera_state.simulation_mode = true;
            camera_state.camera_mgr = nullptr;
            image_width = 1280;
            image_height = 720;
        } else {
            std::cout << "\nInitialized " << num_cameras << " camera(s)" << std::endl;
            camera_state.camera_connected = true;

            // Get camera info
            auto& cam_info = camera_state.camera_mgr->get_camera(0);
            image_width = cam_info.width;
            image_height = cam_info.height;

            std::cout << "Camera: " << cam_info.serial << std::endl;
            std::cout << "Resolution: " << image_width << "x" << image_height << std::endl;
        }
    }

    // Query bias ranges from camera and initialize settings
    std::map<std::string, BiasRange> bias_ranges;

    // Check which monitoring features are supported
    struct MonitoringCapabilities {
        bool has_temperature = false;
        bool has_illumination = false;
        bool has_dead_time = false;
    } monitoring_caps;

    // Only query camera settings if we have a real camera
    if (camera_state.camera_connected) {
        std::cout << "\nDebug: Querying bias settings from camera..." << std::endl;
        auto& cam_info = camera_state.camera_mgr->get_camera(0);
        auto* i_ll_biases = cam_info.camera->get_device().get_facility<Metavision::I_LL_Biases>();
    if (i_ll_biases) {
        std::vector<std::string> bias_names = {"bias_diff", "bias_refr", "bias_fo", "bias_hpf", "bias_pr"};
        for (const auto& name : bias_names) {
            try {
                Metavision::LL_Bias_Info info;
                if (i_ll_biases->get_bias_info(name, info)) {
                    auto range = info.get_bias_range();
                    int current = i_ll_biases->get(name);
                    bias_ranges[name] = {range.first, range.second, current};
                    std::cout << name << " range: [" << range.first << ", "
                             << range.second << "], current: " << current << std::endl;
                } else {
                    std::cout << name << " - not available on this camera" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << name << " - error: " << e.what() << std::endl;
            }
        }

        // Initialize config settings with current camera values (only if available)
        if (bias_ranges.count("bias_diff"))
            config.camera_settings().bias_diff = bias_ranges["bias_diff"].current;
        if (bias_ranges.count("bias_refr"))
            config.camera_settings().bias_refr = bias_ranges["bias_refr"].current;
        if (bias_ranges.count("bias_fo"))
            config.camera_settings().bias_fo = bias_ranges["bias_fo"].current;
        if (bias_ranges.count("bias_hpf"))
            config.camera_settings().bias_hpf = bias_ranges["bias_hpf"].current;
        if (bias_ranges.count("bias_pr"))
            config.camera_settings().bias_pr = bias_ranges["bias_pr"].current;
    }

    // Check monitoring capabilities once at startup
    // Disable monitoring features entirely to avoid error spam
    auto* monitoring_facility = cam_info.camera->get_device().get_facility<Metavision::I_Monitoring>();
    if (monitoring_facility) {
        std::cout << "\nChecking monitoring capabilities..." << std::endl;

        // Temperature
        bool temp_works = false;
        try {
            int temp = monitoring_facility->get_temperature();
            // If we got here without exception and temp is reasonable, mark as supported
            if (temp >= -40 && temp <= 120) {  // Reasonable sensor temp range
                monitoring_caps.has_temperature = true;
                temp_works = true;
                std::cout << "  Temperature: supported (current: " << temp << "°C)" << std::endl;
            }
        } catch (...) {}
        if (!temp_works) {
            std::cout << "  Temperature: not supported" << std::endl;
        }

        // Illumination - FORCE DISABLED due to error spam
        // Even if it "works", it generates HAL errors, so disable it
        monitoring_caps.has_illumination = false;
        std::cout << "  Illumination: disabled (generates errors on this camera)" << std::endl;

        // Pixel Dead Time
        bool deadtime_works = false;
        try {
            int dt = monitoring_facility->get_pixel_dead_time();
            if (dt >= 0 && dt <= 100000) {  // Reasonable dead time range (0-100ms)
                monitoring_caps.has_dead_time = true;
                deadtime_works = true;
                std::cout << "  Pixel Dead Time: supported (current: " << dt << " μs)" << std::endl;
            }
        } catch (...) {}
        if (!deadtime_works) {
            std::cout << "  Pixel Dead Time: not supported" << std::endl;
        }
    }

        std::cout << "Debug: Camera initialization complete!" << std::endl;
    } else {
        // Simulation mode - use default bias ranges
        bias_ranges["bias_diff"] = {-25, 23, 0};
        bias_ranges["bias_refr"] = {-25, 23, 0};
        bias_ranges["bias_fo"] = {-25, 23, 0};
        bias_ranges["bias_hpf"] = {-25, 23, 0};
        bias_ranges["bias_pr"] = {-25, 23, 0};
        std::cout << "Simulation mode: using default bias ranges" << std::endl;
    }

    // Create frame generation algorithm (used in camera mode only)

    if (camera_state.camera_connected) {
        std::cout << "Debug: About to create frame generation algorithm..." << std::endl;
        const uint32_t accumulation_time_us = static_cast<uint32_t>(
            config.camera_settings().accumulation_time_s * 1000000);

        camera_state.frame_gen = std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(
            image_width, image_height, accumulation_time_us);
        std::cout << "Debug: Frame generation algorithm created" << std::endl;

        std::cout << "Frame accumulation time: " << config.camera_settings().accumulation_time_s
                  << "s (" << accumulation_time_us << " us)" << std::endl;

        // Set up frame callback
        camera_state.frame_gen->set_output_callback([](const Metavision::timestamp, cv::Mat& frame) {
            if (!frame.empty()) {
                update_texture(frame);
            }
        });
    }

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

    // Start camera or simulation
    if (camera_state.camera_connected) {
        std::cout << "\nStarting camera..." << std::endl;
        auto& cam_info = camera_state.camera_mgr->get_camera(0);
        cam_info.camera->start();
        std::cout << "Camera started successfully!" << std::endl;
        std::cout << "\nPress ESC or close window to exit\n" << std::endl;

        // Camera event processing thread
        camera_state.event_thread = std::make_unique<std::thread>([&]() {
            auto& cam_info = camera_state.camera_mgr->get_camera(0);
            auto& camera = cam_info.camera;

        // Set up event callback
        camera->cd().add_callback([&](const Metavision::EventCD* begin,
                                     const Metavision::EventCD* end) {
            std::lock_guard<std::mutex> lock(framegen_mutex);
            if (camera_state.frame_gen) {
                camera_state.frame_gen->process_events(begin, end);
            }
        });

            // Process events while running
            while (running && camera->is_running()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    } else {
        // Simulation mode - generate synthetic frames
        std::cout << "\nStarting SIMULATION mode..." << std::endl;
        std::cout << "Generating synthetic event camera frames" << std::endl;
        std::cout << "\nPress ESC or close window to exit\n" << std::endl;

        camera_state.event_thread = std::make_unique<std::thread>([&]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> noise_dist(0, 255);
            std::normal_distribution<float> motion_dist(0.0f, 2.0f);

            float phase = 0.0f;

            while (running) {
                // Generate synthetic frame with some motion/patterns
                cv::Mat sim_frame(image_height, image_width, CV_8UC3);

                // Create interesting patterns that change over time
                for (int y = 0; y < image_height; ++y) {
                    for (int x = 0; x < image_width; ++x) {
                        float fx = static_cast<float>(x) / image_width;
                        float fy = static_cast<float>(y) / image_height;

                        // Animated diagonal stripes
                        float value = (std::sin((fx + fy + phase) * 10.0f) + 1.0f) * 0.5f;

                        // Add some noise for "event" effect
                        value += noise_dist(gen) / 1000.0f;

                        uint8_t intensity = static_cast<uint8_t>(value * 255);
                        sim_frame.at<cv::Vec3b>(y, x) = cv::Vec3b(intensity, intensity, intensity);
                    }
                }

                // Add some moving circles to simulate objects
                int cx = static_cast<int>((std::sin(phase * 2.0f) * 0.3f + 0.5f) * image_width);
                int cy = static_cast<int>((std::cos(phase * 1.5f) * 0.3f + 0.5f) * image_height);
                cv::circle(sim_frame, cv::Point(cx, cy), 50, cv::Scalar(255, 255, 255), -1);

                phase += 0.02f;

                // Update the frame
                update_texture(sim_frame);

                // Sleep to simulate frame rate
                std::this_thread::sleep_for(std::chrono::milliseconds(33));  // ~30 FPS
            }
        });
    }

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
        ImGui::SetNextWindowSize(ImVec2(420, 850), ImGuiCond_FirstUseEver);

        if (ImGui::Begin("Camera Settings")) {
            // Get camera info if connected
            CameraManager::CameraInfo* cam_info_ptr = nullptr;
            if (camera_state.camera_connected && camera_state.camera_mgr) {
                cam_info_ptr = &camera_state.camera_mgr->get_camera(0);
            }

            if (camera_state.camera_connected && cam_info_ptr) {
                ImGui::Text("Camera: %s", cam_info_ptr->serial.c_str());
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Mode: SIMULATION");
                if (ImGui::Button("Connect Camera")) {
                    if (try_connect_camera(config, bias_ranges)) {
                        std::cout << "Successfully connected to camera!" << std::endl;
                    } else {
                        std::cout << "No cameras found or connection failed" << std::endl;
                    }
                }
            }
            ImGui::Text("Resolution: %dx%d", image_width, image_height);
            ImGui::Text("FPS: %.1f", io.Framerate);
            ImGui::Separator();

            ImGui::Text("Camera Biases");
            ImGui::Text("Adjust these to tune event detection");
            ImGui::Spacing();

            auto& cam_settings = config.camera_settings();

            // Use actual camera bias ranges
            if (bias_ranges.count("bias_diff")) {
                auto& range = bias_ranges["bias_diff"];
                if (ImGui::SliderInt("bias_diff", &cam_settings.bias_diff, range.min, range.max)) {
                    settings_changed = true;
                }
                ImGui::TextWrapped("Event detection threshold [%d, %d]", range.min, range.max);
            }
            ImGui::Spacing();

            if (bias_ranges.count("bias_refr")) {
                auto& range = bias_ranges["bias_refr"];
                if (ImGui::SliderInt("bias_refr", &cam_settings.bias_refr, range.min, range.max)) {
                    settings_changed = true;
                }
                ImGui::TextWrapped("Refractory period [%d, %d]", range.min, range.max);
            }
            ImGui::Spacing();

            if (bias_ranges.count("bias_fo")) {
                auto& range = bias_ranges["bias_fo"];
                if (ImGui::SliderInt("bias_fo", &cam_settings.bias_fo, range.min, range.max)) {
                    settings_changed = true;
                }
                ImGui::TextWrapped("Photoreceptor follower [%d, %d]", range.min, range.max);
            }
            ImGui::Spacing();

            if (bias_ranges.count("bias_hpf")) {
                auto& range = bias_ranges["bias_hpf"];
                if (ImGui::SliderInt("bias_hpf", &cam_settings.bias_hpf, range.min, range.max)) {
                    settings_changed = true;
                }
                ImGui::TextWrapped("High-pass filter [%d, %d]", range.min, range.max);
            }
            ImGui::Spacing();

            if (bias_ranges.count("bias_pr")) {
                auto& range = bias_ranges["bias_pr"];
                if (ImGui::SliderInt("bias_pr", &cam_settings.bias_pr, range.min, range.max)) {
                    settings_changed = true;
                }
                ImGui::TextWrapped("Pixel photoreceptor [%d, %d]", range.min, range.max);
            }
            ImGui::Spacing();

            ImGui::Separator();
            ImGui::Text("Frame Generation");

            if (ImGui::SliderFloat("Accumulation (s)", &cam_settings.accumulation_time_s,
                                  0.001f, 0.1f, "%.3f")) {
                settings_changed = true;
            }
            ImGui::TextWrapped("Time to accumulate events into frame");

            // Apply button
            ImGui::Spacing();
            ImGui::Separator();
            if (settings_changed) {
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "Settings changed!");
                if (ImGui::Button("Apply Settings", ImVec2(-1, 0))) {
                    // Apply bias settings
                    if (camera_state.camera_connected && cam_info_ptr) {
                        apply_bias_settings(*cam_info_ptr->camera, cam_settings);
                    }

                    // Update frame generation if accumulation time changed
                    if (cam_settings.accumulation_time_s != previous_settings.accumulation_time_s) {
                        std::cout << "Updating frame accumulation time to "
                                  << cam_settings.accumulation_time_s << "s" << std::endl;

                        const uint32_t new_time_us = static_cast<uint32_t>(
                            cam_settings.accumulation_time_s * 1000000);

                        // Lock mutex to safely recreate frame generator
                        {
                            std::lock_guard<std::mutex> lock(framegen_mutex);

                            // Recreate frame generator with new timing
                            camera_state.frame_gen = std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(
                                image_width, image_height, new_time_us);
                            camera_state.frame_gen->set_output_callback([](const Metavision::timestamp, cv::Mat& frame) {
                                if (!frame.empty()) {
                                    update_texture(frame);
                                }
                            });
                        }

                        std::cout << "Frame generator updated" << std::endl;
                    }

                    previous_settings = cam_settings;
                    settings_changed = false;
                }
            }

            // Reset button
            if (ImGui::Button("Reset to Defaults", ImVec2(-1, 0))) {
                // Reset to middle of each bias range
                if (bias_ranges.count("bias_diff"))
                    cam_settings.bias_diff = (bias_ranges["bias_diff"].min + bias_ranges["bias_diff"].max) / 2;
                if (bias_ranges.count("bias_refr"))
                    cam_settings.bias_refr = (bias_ranges["bias_refr"].min + bias_ranges["bias_refr"].max) / 2;
                if (bias_ranges.count("bias_fo"))
                    cam_settings.bias_fo = (bias_ranges["bias_fo"].min + bias_ranges["bias_fo"].max) / 2;
                if (bias_ranges.count("bias_hpf"))
                    cam_settings.bias_hpf = (bias_ranges["bias_hpf"].min + bias_ranges["bias_hpf"].max) / 2;
                if (bias_ranges.count("bias_pr"))
                    cam_settings.bias_pr = (bias_ranges["bias_pr"].min + bias_ranges["bias_pr"].max) / 2;
                cam_settings.accumulation_time_s = 0.01f;
                settings_changed = true;
            }

            // ===================================================================
            // ADVANCED FEATURES
            // ===================================================================
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Advanced Features");

            // Hardware Monitoring
            Metavision::I_Monitoring* monitoring = nullptr;
            if (camera_state.camera_connected && cam_info_ptr) {
                monitoring = cam_info_ptr->camera->get_device().get_facility<Metavision::I_Monitoring>();
            }
            if (monitoring && (monitoring_caps.has_temperature || monitoring_caps.has_illumination || monitoring_caps.has_dead_time)) {
                if (ImGui::CollapsingHeader("Hardware Monitoring", ImGuiTreeNodeFlags_DefaultOpen)) {
                    if (monitoring_caps.has_temperature) {
                        try {
                            int temp = monitoring->get_temperature();
                            ImGui::Text("Temperature: %d°C", temp);
                            if (temp > 60) {
                                ImGui::SameLine();
                                ImGui::TextColored(ImVec4(1, 0, 0, 1), "⚠ HOT");
                            }
                        } catch (...) {
                            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Temperature: Error");
                        }
                    }

                    if (monitoring_caps.has_illumination) {
                        try {
                            int illum = monitoring->get_illumination();
                            ImGui::Text("Illumination: %d lux", illum);
                        } catch (...) {
                            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Illumination: Error");
                        }
                    }

                    if (monitoring_caps.has_dead_time) {
                        try {
                            int dead_time = monitoring->get_pixel_dead_time();
                            ImGui::Text("Pixel Dead Time: %d μs", dead_time);
                        } catch (...) {
                            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Pixel Dead Time: Error");
                        }
                    }

                    if (!monitoring_caps.has_temperature && !monitoring_caps.has_illumination && !monitoring_caps.has_dead_time) {
                        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "No monitoring features available");
                    }
                }
            }

            // ROI (Region of Interest)
            Metavision::I_ROI* roi = nullptr;
            if (camera_state.camera_connected && cam_info_ptr) {
                roi = cam_info_ptr->camera->get_device().get_facility<Metavision::I_ROI>();
            }
            if (roi) {
                if (ImGui::CollapsingHeader("Region of Interest (ROI)")) {
                    static bool roi_enabled = false;
                    static bool crop_view = false;
                    static int roi_mode = 0;  // 0=ROI, 1=RONI
                    static int roi_x = 0;
                    static int roi_y = 0;
                    static int roi_width = image_width / 2;
                    static int roi_height = image_height / 2;
                    static bool roi_window_changed = false;

                    ImGui::TextWrapped("Define a rectangular region to process or ignore events");
                    ImGui::Spacing();

                    if (ImGui::Checkbox("Enable ROI", &roi_enabled)) {
                        roi->enable(roi_enabled);
                        std::cout << "ROI " << (roi_enabled ? "enabled" : "disabled") << std::endl;

                        // If enabling, apply current window
                        if (roi_enabled) {
                            roi_width = std::min(roi_width, image_width - roi_x);
                            roi_height = std::min(roi_height, image_height - roi_y);
                            Metavision::I_ROI::Window window(roi_x, roi_y, roi_width, roi_height);
                            roi->set_window(window);
                        }
                    }

                    ImGui::SameLine();
                    ImGui::Checkbox("Crop View to ROI", &crop_view);

                    ImGui::Spacing();
                    const char* modes[] = { "ROI (Keep Inside)", "RONI (Discard Inside)" };
                    if (ImGui::Combo("Mode", &roi_mode, modes, 2)) {
                        roi->set_mode(roi_mode == 0 ? Metavision::I_ROI::Mode::ROI : Metavision::I_ROI::Mode::RONI);
                        std::cout << "ROI mode set to " << modes[roi_mode] << std::endl;
                    }

                    ImGui::Spacing();
                    ImGui::Text("Window Position & Size:");

                    // Track if any slider changed
                    roi_window_changed = false;
                    roi_window_changed |= ImGui::SliderInt("X", &roi_x, 0, image_width - 1);
                    roi_window_changed |= ImGui::SliderInt("Y", &roi_y, 0, image_height - 1);
                    roi_window_changed |= ImGui::SliderInt("Width", &roi_width, 1, image_width);
                    roi_window_changed |= ImGui::SliderInt("Height", &roi_height, 1, image_height);

                    // Update visualization in real-time (always, even if ROI not enabled)
                    {
                        std::lock_guard<std::mutex> roi_lock(roi_viz.mutex);
                        roi_viz.show = roi_enabled;
                        roi_viz.crop_to_roi = crop_view;
                        roi_viz.x = roi_x;
                        roi_viz.y = roi_y;
                        roi_viz.width = roi_width;
                        roi_viz.height = roi_height;
                    }

                    // Auto-apply if ROI is enabled and sliders changed
                    if (roi_window_changed && roi_enabled) {
                        // Clamp values
                        roi_width = std::min(roi_width, image_width - roi_x);
                        roi_height = std::min(roi_height, image_height - roi_y);

                        Metavision::I_ROI::Window window(roi_x, roi_y, roi_width, roi_height);
                        roi->set_window(window);

                        // Debug output
                        std::cout << "[ROI UPDATE] Window set to: x=" << roi_x << " y=" << roi_y
                                  << " w=" << roi_width << " h=" << roi_height << std::endl;
                    }

                    ImGui::TextWrapped("Window: [%d, %d] %dx%d", roi_x, roi_y, roi_width, roi_height);
                    if (roi_enabled) {
                        ImGui::SameLine();
                        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "ACTIVE");
                    }
                }
            }

            // ERC (Event Rate Controller)
            Metavision::I_ErcModule* erc = nullptr;
            if (camera_state.camera_connected && cam_info_ptr) {
                erc = cam_info_ptr->camera->get_device().get_facility<Metavision::I_ErcModule>();
            }
            if (erc) {
                if (ImGui::CollapsingHeader("Event Rate Controller (ERC)")) {
                    static bool erc_enabled = false;
                    static int erc_rate_kev = 1000;  // kilo-events per second

                    ImGui::TextWrapped("Limit the maximum event rate to prevent bandwidth saturation");
                    ImGui::Spacing();

                    if (ImGui::Checkbox("Enable ERC", &erc_enabled)) {
                        erc->enable(erc_enabled);
                        std::cout << "ERC " << (erc_enabled ? "enabled" : "disabled") << std::endl;
                    }

                    ImGui::Spacing();
                    uint32_t min_rate = erc->get_min_supported_cd_event_rate() / 1000;  // Convert to kev/s
                    uint32_t max_rate = erc->get_max_supported_cd_event_rate() / 1000;

                    if (ImGui::SliderInt("Event Rate (kev/s)", &erc_rate_kev, min_rate, max_rate)) {
                        uint32_t rate_ev_s = erc_rate_kev * 1000;  // Convert to ev/s
                        erc->set_cd_event_rate(rate_ev_s);
                        std::cout << "ERC rate set to " << rate_ev_s << " ev/s" << std::endl;
                    }

                    ImGui::TextWrapped("Current: %d kev/s (%d Mev/s)", erc_rate_kev, erc_rate_kev / 1000);
                    ImGui::TextWrapped("Range: %d - %d kev/s", min_rate, max_rate);

                    uint32_t period = erc->get_count_period();
                    ImGui::Text("Count Period: %d μs", period);
                }
            }

            // Anti-Flicker Module
            Metavision::I_AntiFlickerModule* antiflicker = nullptr;
            if (camera_state.camera_connected && cam_info_ptr) {
                antiflicker = cam_info_ptr->camera->get_device().get_facility<Metavision::I_AntiFlickerModule>();
            }
            if (antiflicker) {
                if (ImGui::CollapsingHeader("Anti-Flicker Filter")) {
                    static bool af_enabled = false;
                    static int af_mode = 0;  // 0=BAND_STOP, 1=BAND_PASS
                    static int af_low_freq = 100;
                    static int af_high_freq = 150;
                    static int af_duty_cycle = 50;

                    ImGui::TextWrapped("Filter out flicker from artificial lighting (50/60Hz)");
                    ImGui::Spacing();

                    if (ImGui::Checkbox("Enable Anti-Flicker", &af_enabled)) {
                        antiflicker->enable(af_enabled);
                        std::cout << "Anti-Flicker " << (af_enabled ? "enabled" : "disabled") << std::endl;
                    }

                    ImGui::Spacing();

                    // Filter Mode
                    const char* af_modes[] = { "BAND_STOP (Remove frequencies)", "BAND_PASS (Keep frequencies)" };
                    if (ImGui::Combo("Filter Mode", &af_mode, af_modes, 2)) {
                        antiflicker->set_filtering_mode(af_mode == 0 ?
                            Metavision::I_AntiFlickerModule::BAND_STOP :
                            Metavision::I_AntiFlickerModule::BAND_PASS);
                        std::cout << "Anti-Flicker mode set to " << af_modes[af_mode] << std::endl;
                    }

                    ImGui::Spacing();
                    ImGui::Text("Frequency Band:");

                    // Get supported frequency range
                    uint32_t min_freq = antiflicker->get_min_supported_frequency();
                    uint32_t max_freq = antiflicker->get_max_supported_frequency();

                    bool freq_changed = false;
                    freq_changed |= ImGui::SliderInt("Low Frequency (Hz)", &af_low_freq, min_freq, max_freq);
                    freq_changed |= ImGui::SliderInt("High Frequency (Hz)", &af_high_freq, min_freq, max_freq);

                    if (freq_changed) {
                        // Ensure low < high
                        if (af_low_freq >= af_high_freq) {
                            af_high_freq = af_low_freq + 1;
                        }
                        try {
                            antiflicker->set_frequency_band(af_low_freq, af_high_freq);
                            std::cout << "Anti-Flicker frequency band set to [" << af_low_freq
                                     << ", " << af_high_freq << "] Hz" << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error setting frequency band: " << e.what() << std::endl;
                        }
                    }

                    ImGui::Spacing();

                    // Duty Cycle
                    if (ImGui::SliderInt("Duty Cycle (%)", &af_duty_cycle, 0, 100)) {
                        try {
                            antiflicker->set_duty_cycle(af_duty_cycle);
                            std::cout << "Anti-Flicker duty cycle set to " << af_duty_cycle << "%" << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error setting duty cycle: " << e.what() << std::endl;
                        }
                    }

                    ImGui::Spacing();
                    ImGui::TextWrapped("Common presets:");
                    ImGui::SameLine();
                    if (ImGui::SmallButton("50Hz")) {
                        af_low_freq = std::max((int)min_freq, 45);
                        af_high_freq = std::min((int)max_freq, 55);
                        try {
                            antiflicker->set_frequency_band(af_low_freq, af_high_freq);
                            std::cout << "Preset: 50Hz filter set [" << af_low_freq << ", " << af_high_freq << "]" << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error setting 50Hz preset: " << e.what() << std::endl;
                        }
                    }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("60Hz")) {
                        af_low_freq = std::max((int)min_freq, 55);
                        af_high_freq = std::min((int)max_freq, 65);
                        try {
                            antiflicker->set_frequency_band(af_low_freq, af_high_freq);
                            std::cout << "Preset: 60Hz filter set [" << af_low_freq << ", " << af_high_freq << "]" << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error setting 60Hz preset: " << e.what() << std::endl;
                        }
                    }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("100Hz")) {
                        af_low_freq = std::max((int)min_freq, 95);
                        af_high_freq = std::min((int)max_freq, 105);
                        try {
                            antiflicker->set_frequency_band(af_low_freq, af_high_freq);
                            std::cout << "Preset: 100Hz filter set [" << af_low_freq << ", " << af_high_freq << "]" << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error setting 100Hz preset: " << e.what() << std::endl;
                        }
                    }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("120Hz")) {
                        af_low_freq = std::max((int)min_freq, 115);
                        af_high_freq = std::min((int)max_freq, 125);
                        try {
                            antiflicker->set_frequency_band(af_low_freq, af_high_freq);
                            std::cout << "Preset: 120Hz filter set [" << af_low_freq << ", " << af_high_freq << "]" << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error setting 120Hz preset: " << e.what() << std::endl;
                        }
                    }

                    ImGui::TextWrapped("Range: %d - %d Hz", min_freq, max_freq);
                }
            }

            // Event Trail Filter Module
            Metavision::I_EventTrailFilterModule* trail_filter = nullptr;
            if (camera_state.camera_connected && cam_info_ptr) {
                trail_filter = cam_info_ptr->camera->get_device().get_facility<Metavision::I_EventTrailFilterModule>();
            }
            if (trail_filter) {
                if (ImGui::CollapsingHeader("Event Trail Filter")) {
                    static bool etf_enabled = false;
                    static int etf_type = 0;  // 0=TRAIL, 1=STC_CUT_TRAIL, 2=STC_KEEP_TRAIL
                    static int etf_threshold = 10000;  // microseconds

                    ImGui::TextWrapped("Filter noise from event bursts and rapid flickering");
                    ImGui::Spacing();

                    if (ImGui::Checkbox("Enable Trail Filter", &etf_enabled)) {
                        try {
                            trail_filter->enable(etf_enabled);
                            std::cout << "Event Trail Filter " << (etf_enabled ? "enabled" : "disabled") << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error enabling trail filter: " << e.what() << std::endl;
                        }
                    }

                    ImGui::Spacing();

                    // Filter Type
                    const char* etf_types[] = {
                        "TRAIL (Keep first event)",
                        "STC_CUT_TRAIL (Keep second event)",
                        "STC_KEEP_TRAIL (Keep trailing events)"
                    };
                    if (ImGui::Combo("Filter Type", &etf_type, etf_types, 3)) {
                        try {
                            Metavision::I_EventTrailFilterModule::Type type;
                            switch (etf_type) {
                                case 0: type = Metavision::I_EventTrailFilterModule::Type::TRAIL; break;
                                case 1: type = Metavision::I_EventTrailFilterModule::Type::STC_CUT_TRAIL; break;
                                case 2: type = Metavision::I_EventTrailFilterModule::Type::STC_KEEP_TRAIL; break;
                                default: type = Metavision::I_EventTrailFilterModule::Type::TRAIL; break;
                            }
                            trail_filter->set_type(type);
                            std::cout << "Trail filter type set to " << etf_types[etf_type] << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error setting trail filter type: " << e.what() << std::endl;
                        }
                    }

                    ImGui::Spacing();

                    // Threshold Delay
                    ImGui::Text("Threshold Delay:");
                    try {
                        uint32_t min_thresh = trail_filter->get_min_supported_threshold();
                        uint32_t max_thresh = trail_filter->get_max_supported_threshold();

                        if (ImGui::SliderInt("Threshold (μs)", &etf_threshold, min_thresh, max_thresh)) {
                            try {
                                trail_filter->set_threshold(etf_threshold);
                                std::cout << "Trail filter threshold set to " << etf_threshold << " μs" << std::endl;
                            } catch (const std::exception& e) {
                                std::cerr << "Error setting threshold: " << e.what() << std::endl;
                            }
                        }

                        ImGui::TextWrapped("Range: %d - %d μs", min_thresh, max_thresh);
                        ImGui::Spacing();
                        ImGui::TextWrapped("Lower threshold = more aggressive filtering");
                    } catch (const std::exception& e) {
                        std::cerr << "Error getting threshold range: " << e.what() << std::endl;
                        ImGui::TextWrapped("Error: Could not get threshold range");
                    }
                }
            }

            // Digital Crop Module
            Metavision::I_DigitalCrop* digital_crop = nullptr;
            if (camera_state.camera_connected && cam_info_ptr) {
                digital_crop = cam_info_ptr->camera->get_device().get_facility<Metavision::I_DigitalCrop>();
            }
            if (digital_crop) {
                if (ImGui::CollapsingHeader("Digital Crop")) {
                    static bool dc_enabled = false;
                    static int dc_x = 0, dc_y = 0;
                    static int dc_width = image_width;
                    static int dc_height = image_height;

                    ImGui::TextWrapped("Crop sensor output to reduce resolution and data volume");
                    ImGui::Spacing();

                    if (ImGui::Checkbox("Enable Digital Crop", &dc_enabled)) {
                        try {
                            digital_crop->enable(dc_enabled);
                            std::cout << "Digital Crop " << (dc_enabled ? "enabled" : "disabled") << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error enabling digital crop: " << e.what() << std::endl;
                        }
                    }

                    ImGui::Spacing();

                    // Position and size controls
                    ImGui::Text("Crop Region:");
                    bool dc_changed = false;
                    dc_changed |= ImGui::SliderInt("X Position", &dc_x, 0, image_width - 1);
                    dc_changed |= ImGui::SliderInt("Y Position", &dc_y, 0, image_height - 1);
                    dc_changed |= ImGui::SliderInt("Width", &dc_width, 1, image_width);
                    dc_changed |= ImGui::SliderInt("Height", &dc_height, 1, image_height);

                    // Apply crop region if changed and enabled
                    if (dc_changed && dc_enabled) {
                        try {
                            // Ensure crop region is within bounds
                            int max_w = image_width - dc_x;
                            int max_h = image_height - dc_y;
                            dc_width = std::min(dc_width, max_w);
                            dc_height = std::min(dc_height, max_h);

                            // Convert from (x, y, width, height) to (start_x, start_y, end_x, end_y)
                            uint32_t start_x = dc_x;
                            uint32_t start_y = dc_y;
                            uint32_t end_x = dc_x + dc_width - 1;
                            uint32_t end_y = dc_y + dc_height - 1;

                            Metavision::I_DigitalCrop::Region region(start_x, start_y, end_x, end_y);
                            digital_crop->set_window_region(region, false);
                            std::cout << "Digital crop region set to [" << start_x << ", " << start_y
                                     << ", " << end_x << ", " << end_y << "]" << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error setting crop region: " << e.what() << std::endl;
                        }
                    }

                    ImGui::Spacing();
                    ImGui::TextWrapped("Note: Digital crop reduces sensor resolution");
                    ImGui::TextWrapped("Similar to ROI but less flexible");
                }
            }
        }
        ImGui::End();

        // Camera view window
        ImGui::SetNextWindowPos(ImVec2(440, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(window_width - 450, window_height - 20),
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

    // Stop camera if connected
    if (camera_state.camera_connected && camera_state.camera_mgr) {
        auto& cam_info = camera_state.camera_mgr->get_camera(0);
        if (cam_info.camera && cam_info.camera->is_running()) {
            cam_info.camera->stop();
        }
    }

    // Wait for event thread
    if (camera_state.event_thread && camera_state.event_thread->joinable()) {
        camera_state.event_thread->join();
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
