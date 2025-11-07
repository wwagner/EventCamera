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
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>

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

// Camera features
#include "camera/features/erc_feature.h"
#include "camera/features/antiflicker_feature.h"
#include "camera/features/trail_filter_feature.h"
#include "camera/features/digital_crop_feature.h"
#include "camera/features/roi_feature.h"
#include "camera/features/monitoring_feature.h"

// Application state module
#include "core/app_state.h"

// UI modules
#include "ui/settings_panel.h"
#include "camera/bias_manager.h"

// Global application state (replaces all previous global state variables)
std::unique_ptr<core::AppState> app_state;

// Thread synchronization
std::mutex framegen_mutex;  // Protects frame generator from race conditions

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

/**
 * Store frame in frame buffer
 * Implements frame dropping to prevent queue buildup and maintain real-time display
 */
void update_texture(const cv::Mat& frame) {
    if (frame.empty() || !app_state) return;
    app_state->frame_buffer().store_frame(frame);
}

/**
 * Upload OpenCV frame to OpenGL texture
 */
void upload_frame_to_gpu() {
    if (!app_state) return;

    // Try to consume a frame from the buffer
    auto frame_opt = app_state->frame_buffer().consume_frame();
    if (!frame_opt.has_value()) {
        return;  // No new frame available
    }

    // Process frame through filter pipeline
    cv::Mat processed_frame = app_state->frame_processor().process(frame_opt.value());

    // Upload to GPU texture
    app_state->texture_manager().upload_frame(processed_frame);
}

/**
 * Apply camera bias settings
 */
void apply_bias_settings(Metavision::Camera& camera, const AppConfig::CameraSettings& settings) {
    auto* i_ll_biases = camera.get_device().get_facility<Metavision::I_LL_Biases>();
    if (i_ll_biases) {
        std::cout << "Applying camera biases..." << std::endl;

        try {
            i_ll_biases->set("bias_diff", settings.bias_diff);
            std::cout << "  bias_diff=" << settings.bias_diff << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  Warning: Could not set bias_diff: " << e.what() << std::endl;
        }

        try {
            i_ll_biases->set("bias_refr", settings.bias_refr);
            std::cout << "  bias_refr=" << settings.bias_refr << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  Warning: Could not set bias_refr: " << e.what() << std::endl;
        }

        try {
            i_ll_biases->set("bias_fo", settings.bias_fo);
            std::cout << "  bias_fo=" << settings.bias_fo << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  Warning: Could not set bias_fo: " << e.what() << std::endl;
        }

        try {
            i_ll_biases->set("bias_hpf", settings.bias_hpf);
            std::cout << "  bias_hpf=" << settings.bias_hpf << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  Warning: Could not set bias_hpf: " << e.what() << std::endl;
        }

        std::cout << "Camera biases applied successfully" << std::endl;
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
    if (app_state && app_state->camera_state().is_connected() && app_state->camera_state().camera_manager()) {
        auto& cam_info = app_state->camera_state().camera_manager()->get_camera(0);
        auto* i_ll_biases = cam_info.camera->get_device().get_facility<Metavision::I_LL_Biases>();

        if (i_ll_biases) {
            try {
                i_ll_biases->set("bias_diff", genome.bias_diff);
                i_ll_biases->set("bias_refr", genome.bias_refr);
                i_ll_biases->set("bias_fo", genome.bias_fo);
                i_ll_biases->set("bias_hpf", genome.bias_hpf);
            } catch (const std::exception& e) {
                std::cerr << "Error applying biases: " << e.what() << std::endl;
                result.combined_fitness = 1e9f;  // Very bad fitness
                return result;
            }
        }

        // Update accumulation time
        if (app_state->camera_state().frame_generator()) {
            const uint32_t accumulation_time_us = static_cast<uint32_t>(
                genome.accumulation_time_s * 1000000);

            std::lock_guard<std::mutex> lock(framegen_mutex);
            int width = app_state->display_settings().get_image_width();
            int height = app_state->display_settings().get_image_height();
            app_state->camera_state().frame_generator() = std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(
                width, height, accumulation_time_us);
            app_state->camera_state().frame_generator()->set_output_callback([](const Metavision::timestamp ts, cv::Mat& frame) {
                if (frame.empty() || !app_state) return;

                // Rate limit display updates to target FPS
                auto now = std::chrono::steady_clock::now();
                auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
                int fps_target = app_state->display_settings().get_target_fps();

                // Check if enough time has passed since last display
                if (app_state->frame_sync().should_display_frame(now_us, fps_target)) {
                    app_state->frame_sync().on_frame_generated(ts, now_us);
                    app_state->frame_sync().on_frame_displayed(now_us);
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
    
    // Wait for parameters to stabilize (reduced for faster GA)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Capture frames
    std::vector<cv::Mat> captured_frames;
    captured_frames.reserve(num_frames);
    
    for (int i = 0; i < num_frames; ++i) {
        // Try to get a frame from the buffer
        if (app_state) {
            auto frame_opt = app_state->frame_buffer().consume_frame();
            if (frame_opt.has_value() && !frame_opt.value().empty()) {
                captured_frames.push_back(frame_opt.value());
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));  // Fast frame capture for GA
    }
    
    // Calculate fitness metrics from captured frames
    if (captured_frames.empty()) {
        result.combined_fitness = 1e9f;  // Very bad fitness
        return result;
    }

    result.num_valid_frames = captured_frames.size();
    result.total_frames = num_frames;

    // Calculate contrast (average across all frames)
    float total_contrast = 0.0f;
    for (const auto& frame : captured_frames) {
        total_contrast += EventCameraGeneticOptimizer::calculate_contrast(frame);
    }
    result.contrast_score = total_contrast / captured_frames.size();

    // Calculate noise metrics
    result.temporal_variance = EventCameraGeneticOptimizer::calculate_temporal_variance(captured_frames);
    result.spatial_noise = EventCameraGeneticOptimizer::calculate_spatial_noise(captured_frames[0]);
    result.noise_metric = result.temporal_variance + result.spatial_noise;

    // Calculate isolated pixel ratio (single-pixel noise)
    result.isolated_pixel_ratio = EventCameraGeneticOptimizer::calculate_isolated_pixels(captured_frames[0]);

    // Calculate total event pixels (bright pixels above threshold)
    cv::Mat gray;
    if (captured_frames[0].channels() == 3) {
        cv::cvtColor(captured_frames[0], gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = captured_frames[0];
    }
    cv::Mat binary;
    cv::threshold(gray, binary, 10, 255, cv::THRESH_BINARY);
    result.total_event_pixels = cv::countNonZero(binary);

    // Calculate mean brightness
    cv::Scalar mean_val = cv::mean(captured_frames[0]);
    result.mean_brightness = static_cast<float>(mean_val[0]);

    // Combined fitness will be calculated by optimizer using alpha/beta weights
    result.combined_fitness = 0.0f;  // Will be set by optimizer

    return result;
}


/**
 * Attempt to connect to a camera (for hot-plug support)
 */
bool try_connect_camera(AppConfig& config, EventCamera::BiasManager& bias_mgr,
                        const std::string& serial_hint = "") {
    if (!app_state) return false;

    std::lock_guard<std::mutex> lock(app_state->camera_state().connection_mutex());

    if (app_state->camera_state().is_connected()) {
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
    if (app_state->camera_state().is_simulation_mode() && app_state->camera_state().event_thread()) {
        app_state->request_shutdown();
        if (app_state->camera_state().event_thread()->joinable()) {
            app_state->camera_state().event_thread()->join();
        }
        // Reset running state
        app_state = std::make_unique<core::AppState>();
    }

    // Initialize camera manager
    app_state->camera_state().camera_manager() = std::make_unique<CameraManager>();
    int num_cameras = app_state->camera_state().camera_manager()->initialize(serial_hint);

    if (num_cameras == 0) {
        std::cerr << "Failed to initialize camera" << std::endl;
        app_state->camera_state().camera_manager() = nullptr;
        return false;
    }

    auto& cam_info = app_state->camera_state().camera_manager()->get_camera(0);
    app_state->display_settings().set_image_size(cam_info.width, cam_info.height);

    // Initialize BiasManager with newly connected camera
    if (bias_mgr.initialize(*cam_info.camera)) {
        std::cout << "BiasManager initialized with new camera" << std::endl;
    } else {
        std::cerr << "Warning: Failed to initialize BiasManager with new camera" << std::endl;
    }

    // Register and initialize hardware features
    std::cout << "\nRegistering hardware features..." << std::endl;
    app_state->feature_manager().register_feature(std::make_shared<EventCamera::ERCFeature>());
    app_state->feature_manager().register_feature(std::make_shared<EventCamera::AntiFlickerFeature>());
    app_state->feature_manager().register_feature(std::make_shared<EventCamera::TrailFilterFeature>());
    app_state->feature_manager().register_feature(std::make_shared<EventCamera::DigitalCropFeature>(app_state->display_settings()));
    app_state->feature_manager().register_feature(std::make_shared<EventCamera::ROIFeature>(app_state->roi_filter(), app_state->display_settings()));
    app_state->feature_manager().register_feature(std::make_shared<EventCamera::MonitoringFeature>());

    app_state->feature_manager().initialize_all(*cam_info.camera);
    std::cout << "Hardware features initialized\n" << std::endl;

    // Create frame generator
    const uint32_t accumulation_time_us = static_cast<uint32_t>(
        config.camera_settings().accumulation_time_s * 1000000);

    int width = app_state->display_settings().get_image_width();
    int height = app_state->display_settings().get_image_height();
    app_state->camera_state().frame_generator() = std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(
        width, height, accumulation_time_us);

    app_state->camera_state().frame_generator()->set_output_callback([](const Metavision::timestamp ts, cv::Mat& frame) {
        if (frame.empty() || !app_state) return;

        // Rate limit display updates to target FPS
        auto now = std::chrono::steady_clock::now();
        auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
        int fps_target = app_state->display_settings().get_target_fps();

        // Check if enough time has passed since last display
        if (app_state->frame_sync().should_display_frame(now_us, fps_target)) {
            app_state->frame_sync().on_frame_generated(ts, now_us);
            app_state->frame_sync().on_frame_displayed(now_us);
            update_texture(frame);  // Uses FrameBuffer which handles dropping
        }
        // Note: frame_buffer tracks its own dropped/generated statistics
    });

    // Start camera
    auto start_time = std::chrono::steady_clock::now();
    int64_t start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(start_time.time_since_epoch()).count();
    app_state->camera_state().set_camera_start_time_us(start_time_us);

    cam_info.camera->start();
    std::cout << "Camera started: " << cam_info.serial << std::endl;

    // Start camera event thread
    app_state->camera_state().event_thread() = std::make_unique<std::thread>([&]() {
        auto& cam_info = app_state->camera_state().camera_manager()->get_camera(0);
        auto& camera = cam_info.camera;

        camera->cd().add_callback([&](const Metavision::EventCD* begin,
                                     const Metavision::EventCD* end) {
            if (begin == end || !app_state) return;

            // Count events for event rate calculation
            int64_t event_count = end - begin;

            // Track last event timestamp to measure event latency
            Metavision::timestamp last_ts = (end-1)->t;

            // Get current time
            auto now = std::chrono::steady_clock::now();
            auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

            // Record events for metrics
            app_state->event_metrics().record_events(event_count, now_us);
            app_state->event_metrics().record_event_timestamp(last_ts);

            // CRITICAL FIX: Skip ALL old event batches immediately
            // Check age of newest event - if old, skip entire batch without iterating
            int64_t cam_start = app_state->camera_state().get_camera_start_time_us();

            Metavision::timestamp newest_event_ts = (end-1)->t;
            int64_t newest_event_system_ts = cam_start + newest_event_ts;
            int64_t event_age_us = now_us - newest_event_system_ts;

            // Skip any batch where even the newest event is older than 50ms
            if (event_age_us > 50000) {  // 50ms - be aggressive about skipping old data
                return;  // Skip entire batch immediately
            }

            // Only process recent batches
            std::lock_guard<std::mutex> lock(framegen_mutex);
            if (app_state && app_state->camera_state().frame_generator()) {
                app_state->camera_state().frame_generator()->process_events(begin, end);
            }
        });

        while (app_state && app_state->is_running() && camera->is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));  // 0.1ms for low latency
        }
    });
    
    app_state->camera_state().set_connected(true);
    app_state->camera_state().set_simulation_mode(false);

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

    // Initialize application state
    std::cout << "\nInitializing application state..." << std::endl;
    app_state = std::make_unique<core::AppState>();
    std::cout << "Application state initialized" << std::endl;

    // Camera/Simulation Mode

    // List available cameras
    std::cout << "\nScanning for event cameras..." << std::endl;
    auto available_cameras = CameraManager::list_available_cameras();

    if (available_cameras.empty()) {
        std::cout << "No event cameras found - starting in SIMULATION MODE" << std::endl;
        std::cout << "You can connect a camera later using the 'Connect Camera' button" << std::endl;
        app_state->camera_state().set_simulation_mode(true);
        app_state->display_settings().set_image_size(1280, 720);
    } else {
        std::cout << "Found " << available_cameras.size() << " camera(s):" << std::endl;
        for (size_t i = 0; i < available_cameras.size(); ++i) {
            std::cout << "  [" << i << "] " << available_cameras[i] << std::endl;
        }

        // Initialize camera manager
        app_state->camera_state().camera_manager() = std::make_unique<CameraManager>();
        std::string serial1 = available_cameras[0];  // Use first camera
        int num_cameras = app_state->camera_state().camera_manager()->initialize(serial1);

        if (num_cameras == 0) {
            std::cerr << "ERROR: Failed to initialize camera - starting in SIMULATION MODE" << std::endl;
            app_state->camera_state().set_simulation_mode(true);
            app_state->camera_state().camera_manager() = nullptr;
            app_state->display_settings().set_image_size(1280, 720);
        } else {
            std::cout << "\nInitialized " << num_cameras << " camera(s)" << std::endl;
            app_state->camera_state().set_connected(true);

            // Get camera info
            auto& cam_info = app_state->camera_state().camera_manager()->get_camera(0);
            app_state->display_settings().set_image_size(cam_info.width, cam_info.height);

            std::cout << "Camera: " << cam_info.serial << std::endl;
            std::cout << "Resolution: " << cam_info.width << "x" << cam_info.height << std::endl;
        }
    }

    // Initialize BiasManager for camera bias control
    EventCamera::BiasManager bias_manager;

    // Check which monitoring features are supported
    struct MonitoringCapabilities {
        bool has_temperature = false;
        bool has_illumination = false;
        bool has_dead_time = false;
    } monitoring_caps;

    // Initialize BiasManager with camera settings
    if (app_state->camera_state().is_connected()) {
        std::cout << "\nInitializing BiasManager with camera..." << std::endl;
        auto& cam_info = app_state->camera_state().camera_manager()->get_camera(0);
        if (bias_manager.initialize(*cam_info.camera)) {
            std::cout << "BiasManager initialized successfully" << std::endl;

            // Display available bias ranges
            const auto& bias_ranges = bias_manager.get_bias_ranges();
            for (const auto& [name, range] : bias_ranges) {
                std::cout << name << " range: [" << range.min << ", "
                         << range.max << "], current: " << range.current << std::endl;
            }

            // Apply config settings from INI file to camera
            std::cout << "\nApplying bias settings from config file..." << std::endl;
            bias_manager.set_bias("bias_diff", config.camera_settings().bias_diff);
            bias_manager.set_bias("bias_diff_on", config.camera_settings().bias_diff_on);
            bias_manager.set_bias("bias_diff_off", config.camera_settings().bias_diff_off);
            bias_manager.set_bias("bias_refr", config.camera_settings().bias_refr);
            bias_manager.set_bias("bias_fo", config.camera_settings().bias_fo);
            bias_manager.set_bias("bias_hpf", config.camera_settings().bias_hpf);
            bias_manager.apply_to_camera();
            std::cout << "Bias settings applied from config" << std::endl;
        } else {
            std::cerr << "Warning: Failed to initialize BiasManager" << std::endl;
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
        // Simulation mode - setup default bias ranges
        std::cout << "Simulation mode: setting up default bias ranges" << std::endl;
        bias_manager.setup_simulation_defaults();
    }

    // Create frame generation algorithm (used in camera mode only)

    if (app_state->camera_state().is_connected()) {
        std::cout << "Debug: About to create frame generation algorithm..." << std::endl;
        const uint32_t accumulation_time_us = static_cast<uint32_t>(
            config.camera_settings().accumulation_time_s * 1000000);

        int width = app_state->display_settings().get_image_width();
        int height = app_state->display_settings().get_image_height();
        app_state->camera_state().frame_generator() = std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(
            width, height, accumulation_time_us);
        std::cout << "Debug: Frame generation algorithm created" << std::endl;

        std::cout << "Frame accumulation time: " << config.camera_settings().accumulation_time_s
                  << "s (" << accumulation_time_us << " us)" << std::endl;

        // Set up frame callback with FPS limiting
        app_state->camera_state().frame_generator()->set_output_callback([](const Metavision::timestamp ts, cv::Mat& frame) {
            if (frame.empty() || !app_state) return;

            // Rate limit display updates to target FPS
            auto now = std::chrono::steady_clock::now();
            auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
            int fps_target = app_state->display_settings().get_target_fps();

            // Check if enough time has passed since last display
            if (app_state->frame_sync().should_display_frame(now_us, fps_target)) {
                app_state->frame_sync().on_frame_generated(ts, now_us);
                app_state->frame_sync().on_frame_displayed(now_us);
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
    glfwSwapInterval(0); // Disable vsync for lowest latency

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "ERROR: Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Video processing module already initialized by AppState constructor

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Start camera or simulation
    if (app_state->camera_state().is_connected()) {
        std::cout << "\nStarting camera..." << std::endl;
        auto& cam_info = app_state->camera_state().camera_manager()->get_camera(0);
        cam_info.camera->start();
        std::cout << "Camera started successfully!" << std::endl;
        std::cout << "\nPress ESC or close window to exit\n" << std::endl;

        // Camera event processing thread
        app_state->camera_state().event_thread() = std::make_unique<std::thread>([&]() {
            auto& cam_info = app_state->camera_state().camera_manager()->get_camera(0);
            auto& camera = cam_info.camera;

        // Set up event callback
        camera->cd().add_callback([&](const Metavision::EventCD* begin,
                                     const Metavision::EventCD* end) {
            std::lock_guard<std::mutex> lock(framegen_mutex);
            if (app_state && app_state->camera_state().frame_generator()) {
                app_state->camera_state().frame_generator()->process_events(begin, end);
            }
        });

            // Process events while running
            while (app_state && app_state->is_running() && camera->is_running()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    } else {
        // Simulation mode - generate synthetic frames
        std::cout << "\nStarting SIMULATION mode..." << std::endl;
        std::cout << "Generating synthetic event camera frames" << std::endl;
        std::cout << "\nPress ESC or close window to exit\n" << std::endl;

        app_state->camera_state().event_thread() = std::make_unique<std::thread>([&]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> noise_dist(0, 255);
            std::normal_distribution<float> motion_dist(0.0f, 2.0f);

            float phase = 0.0f;

            while (app_state && app_state->is_running()) {
                int width = app_state->display_settings().get_image_width();
                int height = app_state->display_settings().get_image_height();

                // Generate synthetic frame with some motion/patterns
                cv::Mat sim_frame(height, width, CV_8UC3);

                // Create interesting patterns that change over time
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        float fx = static_cast<float>(x) / width;
                        float fy = static_cast<float>(y) / height;

                        // Animated diagonal stripes
                        float value = (std::sin((fx + fy + phase) * 10.0f) + 1.0f) * 0.5f;

                        // Add some noise for "event" effect
                        value += noise_dist(gen) / 1000.0f;

                        uint8_t intensity = static_cast<uint8_t>(value * 255);
                        sim_frame.at<cv::Vec3b>(y, x) = cv::Vec3b(intensity, intensity, intensity);
                    }
                }

                // Add some moving circles to simulate objects
                int cx = static_cast<int>((std::sin(phase * 2.0f) * 0.3f + 0.5f) * width);
                int cy = static_cast<int>((std::cos(phase * 1.5f) * 0.3f + 0.5f) * height);
                cv::circle(sim_frame, cv::Point(cx, cy), 50, cv::Scalar(255, 255, 255), -1);

                phase += 0.02f;

                // Update the frame
                update_texture(sim_frame);

                // Sleep to simulate frame rate
                std::this_thread::sleep_for(std::chrono::milliseconds(33));  // ~30 FPS
            }
        });
    }

    // Create SettingsPanel for UI
    ui::SettingsPanel settings_panel(*app_state, config, bias_manager);

    // ImageJ streaming state
    auto last_stream_time = std::chrono::steady_clock::now();
    int stream_frame_counter = 0;

    // Main render loop
    while (!glfwWindowShouldClose(window) && app_state->is_running()) {
        glfwPollEvents();

        // Handle ESC key
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            app_state->request_shutdown();
        }

        // ImageJ streaming logic
        if (config.camera_settings().imagej_streaming_enabled) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_stream_time).count();
            int interval_ms = 1000 / config.camera_settings().imagej_stream_fps;

            if (elapsed_ms >= interval_ms) {
                cv::Mat frame = app_state->texture_manager().get_last_frame();
                if (!frame.empty()) {
                    // Create stream directory if it doesn't exist
                    std::string stream_dir = config.camera_settings().imagej_stream_directory;
                    std::filesystem::create_directories(stream_dir);

                    // Generate filename with counter
                    std::stringstream ss;
                    ss << stream_dir;
                    if (!stream_dir.empty() && stream_dir.back() != '\\' && stream_dir.back() != '/') {
                        ss << "\\";
                    }
                    ss << "stream_" << std::setfill('0') << std::setw(6) << stream_frame_counter << ".png";
                    std::string filepath = ss.str();

                    // Save frame
                    cv::imwrite(filepath, frame);
                    stream_frame_counter++;

                    // Clean up old files if needed
                    if (stream_frame_counter > config.camera_settings().imagej_max_stream_files) {
                        int old_frame = stream_frame_counter - config.camera_settings().imagej_max_stream_files - 1;
                        std::stringstream old_ss;
                        old_ss << stream_dir;
                        if (!stream_dir.empty() && stream_dir.back() != '\\' && stream_dir.back() != '/') {
                            old_ss << "\\";
                        }
                        old_ss << "stream_" << std::setfill('0') << std::setw(6) << old_frame << ".png";
                        std::filesystem::remove(old_ss.str());
                    }
                }
                last_stream_time = now;
            }
        }

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Render settings panel
        settings_panel.render();

        // Handle camera connection requests from SettingsPanel
        if (settings_panel.camera_reconnect_requested()) {
            settings_panel.reset_camera_reconnect_request();

            std::cout << "\n=== Disconnecting camera ===" << std::endl;

            // Get camera info before disconnecting
            CameraManager::CameraInfo* cam_info_ptr = nullptr;
            if (app_state->camera_state().is_connected() && app_state->camera_state().camera_manager()) {
                cam_info_ptr = &app_state->camera_state().camera_manager()->get_camera(0);

                // Stop camera first
                try {
                    cam_info_ptr->camera->stop();
                    std::cout << "Camera stopped" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Error stopping camera: " << e.what() << std::endl;
                }
            }

            // Stop event thread
            if (app_state->camera_state().event_thread() && app_state->camera_state().event_thread()->joinable()) {
                app_state->request_shutdown();
                std::cout << "Waiting for event thread to stop..." << std::endl;
                app_state->camera_state().event_thread()->join();
                app_state->camera_state().event_thread().reset();
                std::cout << "Event thread stopped" << std::endl;
            }

            // Reset the running flag so camera can start again
            app_state->reset_running_flag();

            // Shutdown and clear hardware features
            app_state->feature_manager().shutdown_all();
            app_state->feature_manager().clear();

            // Clear camera resources
            app_state->camera_state().frame_generator().reset();
            app_state->camera_state().camera_manager().reset();
            app_state->camera_state().set_connected(false);
            app_state->texture_manager().reset();

            std::cout << "Camera disconnected" << std::endl;
            std::cout << "\n=== Reconnecting camera ===" << std::endl;

            // Reconnect
            if (try_connect_camera(config, bias_manager)) {
                std::cout << "Successfully reconnected to camera!" << std::endl;
            } else {
                std::cout << "No cameras found or connection failed" << std::endl;
            }
        }

        if (settings_panel.camera_connect_requested()) {
            settings_panel.reset_camera_connect_request();

            if (try_connect_camera(config, bias_manager)) {
                std::cout << "Successfully connected to camera!" << std::endl;
            } else {
                std::cout << "No cameras found or connection failed" << std::endl;
            }
        }

        // Advanced Features panel (TEMP: Hidden - moved to SettingsPanel)
        /*
        ImGui::SetNextWindowPos(ImVec2(10, 620), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, window_height - 630), ImGuiCond_FirstUseEver);

        if (ImGui::Begin("Camera Settings - Advanced")) {
        */
        if (false) { // TEMP: Disabled
            // Get camera info pointer for Advanced Features section
            CameraManager::CameraInfo* cam_info_ptr = nullptr;
            if (app_state->camera_state().is_connected() && app_state->camera_state().camera_manager()) {
                cam_info_ptr = &app_state->camera_state().camera_manager()->get_camera(0);
            }

            // ===================================================================
            // ADVANCED FEATURES
            // ===================================================================
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Advanced Features");








            // Hardware Monitoring
            Metavision::I_Monitoring* monitoring = nullptr;
            if (app_state->camera_state().is_connected() && cam_info_ptr) {
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
            if (app_state->camera_state().is_connected() && cam_info_ptr) {
                roi = cam_info_ptr->camera->get_device().get_facility<Metavision::I_ROI>();
            }
            if (roi) {
                if (ImGui::CollapsingHeader("Region of Interest (ROI)")) {
                    static bool roi_enabled = false;
                    static bool crop_view = false;
                    static int roi_mode = 0;  // 0=ROI, 1=RONI
                    static int roi_x = 0;
                    static int roi_y = 0;
                    static int roi_width = app_state->display_settings().get_image_width() / 2;
                    static int roi_height = app_state->display_settings().get_image_height() / 2;
                    static bool roi_window_changed = false;

                    ImGui::TextWrapped("Define a rectangular region to process or ignore events");
                    ImGui::Spacing();

                    if (ImGui::Checkbox("Enable ROI", &roi_enabled)) {
                        roi->enable(roi_enabled);
                        std::cout << "ROI " << (roi_enabled ? "enabled" : "disabled") << std::endl;

                        // If enabling, apply current window
                        if (roi_enabled) {
                            roi_width = std::min(roi_width, app_state->display_settings().get_image_width() - roi_x);
                            roi_height = std::min(roi_height, app_state->display_settings().get_image_height() - roi_y);
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
                    roi_window_changed |= ImGui::SliderInt("X", &roi_x, 0, app_state->display_settings().get_image_width() - 1);
                    roi_window_changed |= ImGui::SliderInt("Y", &roi_y, 0, app_state->display_settings().get_image_height() - 1);
                    roi_window_changed |= ImGui::SliderInt("Width", &roi_width, 1, app_state->display_settings().get_image_width());
                    roi_window_changed |= ImGui::SliderInt("Height", &roi_height, 1, app_state->display_settings().get_image_height());

                    // Update visualization in real-time (always, even if ROI not enabled)
                    if (app_state->roi_filter()) {
                        app_state->roi_filter()->set_enabled(roi_enabled);
                        app_state->roi_filter()->set_crop_to_roi(crop_view);
                        app_state->roi_filter()->set_show_rectangle(roi_enabled);
                        app_state->roi_filter()->set_roi(roi_x, roi_y, roi_width, roi_height);
                    }

                    // Auto-apply if ROI is enabled and sliders changed
                    if (roi_window_changed && roi_enabled) {
                        // Clamp values
                        roi_width = std::min(roi_width, app_state->display_settings().get_image_width() - roi_x);
                        roi_height = std::min(roi_height, app_state->display_settings().get_image_height() - roi_y);

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
            if (app_state->camera_state().is_connected() && cam_info_ptr) {
                erc = cam_info_ptr->camera->get_device().get_facility<Metavision::I_ErcModule>();
            }
            if (erc) {
                if (ImGui::CollapsingHeader("Event Rate Controller (ERC)")) {
                    static bool erc_enabled = false;
                    static int erc_rate_kev = 1000;  // kilo-events per second
                    static bool erc_initialized = false;

                    // Sync UI with hardware state on first render
                    if (!erc_initialized) {
                        try {
                            // Note: Metavision SDK doesn't provide a way to read enabled state
                            // We can only read the rate
                            uint32_t current_rate = erc->get_cd_event_rate();
                            erc_rate_kev = current_rate / 1000;
                            std::cout << "ERC: Synced with hardware - current rate: " << current_rate << " ev/s" << std::endl;
                            erc_initialized = true;
                        } catch (const std::exception& e) {
                            std::cerr << "ERC: Could not read initial state: " << e.what() << std::endl;
                            erc_initialized = true;  // Don't retry every frame
                        }
                    }

                    ImGui::TextWrapped("Limit the maximum event rate to prevent bandwidth saturation");
                    ImGui::Spacing();

                    if (ImGui::Checkbox("Enable ERC", &erc_enabled)) {
                        try {
                            erc->enable(erc_enabled);
                            std::cout << "ERC " << (erc_enabled ? "enabled" : "disabled") << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "ERROR enabling ERC: " << e.what() << std::endl;
                            erc_enabled = !erc_enabled;  // Revert checkbox
                        }
                    }

                    ImGui::Spacing();
                    uint32_t min_rate = erc->get_min_supported_cd_event_rate() / 1000;  // Convert to kev/s
                    uint32_t max_rate = erc->get_max_supported_cd_event_rate() / 1000;

                    if (ImGui::SliderInt("Event Rate (kev/s)", &erc_rate_kev, min_rate, max_rate)) {
                        try {
                            uint32_t rate_ev_s = erc_rate_kev * 1000;  // Convert to ev/s
                            erc->set_cd_event_rate(rate_ev_s);
                            std::cout << "ERC rate set to " << rate_ev_s << " ev/s" << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "ERROR setting ERC rate: " << e.what() << std::endl;
                        }
                    }

                    ImGui::TextWrapped("Current: %d kev/s (%d Mev/s)", erc_rate_kev, erc_rate_kev / 1000);
                    ImGui::TextWrapped("Range: %d - %d kev/s", min_rate, max_rate);

                    uint32_t period = erc->get_count_period();
                    ImGui::Text("Count Period: %d μs", period);
                }
            }

            // Anti-Flicker Module
            Metavision::I_AntiFlickerModule* antiflicker = nullptr;
            if (app_state->camera_state().is_connected() && cam_info_ptr) {
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
                        try {
                            antiflicker->enable(af_enabled);
                            std::cout << "Anti-Flicker " << (af_enabled ? "enabled" : "disabled") << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "ERROR enabling Anti-Flicker: " << e.what() << std::endl;
                            af_enabled = !af_enabled;  // Revert checkbox
                        }
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
            if (app_state->camera_state().is_connected() && cam_info_ptr) {
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
            if (app_state->camera_state().is_connected() && cam_info_ptr) {
                digital_crop = cam_info_ptr->camera->get_device().get_facility<Metavision::I_DigitalCrop>();
            }
            if (digital_crop) {
                if (ImGui::CollapsingHeader("Digital Crop")) {
                    static bool dc_enabled = false;
                    static int dc_x = 0, dc_y = 0;
                    static int dc_width = app_state->display_settings().get_image_width();
                    static int dc_height = app_state->display_settings().get_image_height();

                    ImGui::TextWrapped("Crop sensor output to reduce resolution and data volume");
                    ImGui::Spacing();

                    if (ImGui::Checkbox("Enable Digital Crop", &dc_enabled)) {
                        try {
                            digital_crop->enable(dc_enabled);
                            std::cout << "Digital Crop " << (dc_enabled ? "enabled" : "disabled") << std::endl;

                            // Set the crop region when enabling
                            if (dc_enabled) {
                                // Ensure crop region is within bounds
                                int max_w = app_state->display_settings().get_image_width() - dc_x;
                                int max_h = app_state->display_settings().get_image_height() - dc_y;
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
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "Error enabling digital crop: " << e.what() << std::endl;
                            std::cerr << "Note: Some digital features require camera restart to take effect" << std::endl;
                            dc_enabled = !dc_enabled;  // Revert checkbox
                        }
                    }

                    ImGui::Spacing();

                    // Position and size controls
                    ImGui::Text("Crop Region:");
                    bool dc_changed = false;
                    dc_changed |= ImGui::SliderInt("X Position", &dc_x, 0, app_state->display_settings().get_image_width() - 1);
                    dc_changed |= ImGui::SliderInt("Y Position", &dc_y, 0, app_state->display_settings().get_image_height() - 1);
                    dc_changed |= ImGui::SliderInt("Width", &dc_width, 1, app_state->display_settings().get_image_width());
                    dc_changed |= ImGui::SliderInt("Height", &dc_height, 1, app_state->display_settings().get_image_height());

                    // Apply crop region if changed and enabled
                    if (dc_changed && dc_enabled) {
                        try {
                            // Ensure crop region is within bounds
                            int max_w = app_state->display_settings().get_image_width() - dc_x;
                            int max_h = app_state->display_settings().get_image_height() - dc_y;
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

        // Genetic Algorithm Optimization section
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::CollapsingHeader("Genetic Algorithm Optimization", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextWrapped("Optimize camera parameters using genetic algorithm");
            ImGui::Separator();

            auto& ga_cfg = config.ga_settings();

            if (!ga_state.running) {
                ImGui::Text("Configuration (from event_config.ini)");
                ImGui::Text("Population: %d | Generations: %d", ga_cfg.population_size, ga_cfg.num_generations);
                ImGui::Text("Mutation: %.2f | Crossover: %.2f", ga_cfg.mutation_rate, ga_cfg.crossover_rate);
                ImGui::Text("Frames/Eval: %d", ga_cfg.frames_per_eval);

                ImGui::Separator();
                ImGui::Text("Parameters to Optimize:");
                ImGui::Text("(Edit event_config.ini to change)");

                ImGui::BeginDisabled();
                bool opt_bd = ga_cfg.optimize_bias_diff;
                bool opt_br = ga_cfg.optimize_bias_refr;
                bool opt_bf = ga_cfg.optimize_bias_fo;
                bool opt_bh = ga_cfg.optimize_bias_hpf;
                bool opt_ac = ga_cfg.optimize_accumulation;
                bool opt_tf = ga_cfg.optimize_trail_filter;
                bool opt_af = ga_cfg.optimize_antiflicker;
                bool opt_er = ga_cfg.optimize_erc;

                ImGui::Checkbox("bias_diff##ga_opt", &opt_bd); ImGui::SameLine();
                ImGui::Checkbox("bias_refr##ga_opt", &opt_br);
                ImGui::Checkbox("bias_fo##ga_opt", &opt_bf); ImGui::SameLine();
                ImGui::Checkbox("bias_hpf##ga_opt", &opt_bh);
                ImGui::Checkbox("accumulation##ga_opt", &opt_ac);
                ImGui::Checkbox("trail_filter##ga_opt", &opt_tf); ImGui::SameLine();
                ImGui::Checkbox("antiflicker##ga_opt", &opt_af);
                ImGui::Checkbox("erc##ga_opt", &opt_er);
                ImGui::EndDisabled();

                ImGui::Separator();

                if (app_state->camera_state().is_connected()) {
                    if (ImGui::Button("Start Optimization", ImVec2(-1, 0))) {
                        std::cout << "\n=== Starting GA Optimization ===" << std::endl;

                        // Stop any previous optimization
                        if (ga_state.optimizer && ga_state.running) {
                            ga_state.optimizer->stop();
                            if (ga_state.optimizer_thread && ga_state.optimizer_thread->joinable()) {
                                ga_state.optimizer_thread->join();
                            }
                        }

                        // Reset GA state
                        ga_state.running = true;
                        ga_state.current_generation = 0;
                        ga_state.best_fitness = 1e9f;

                        // Setup optimizer parameters from config
                        EventCameraGeneticOptimizer::OptimizerParams params;
                        params.population_size = ga_cfg.population_size;
                        params.num_generations = ga_cfg.num_generations;
                        params.mutation_rate = ga_cfg.mutation_rate;
                        params.crossover_rate = ga_cfg.crossover_rate;
                        // Fitness weights - tune these for your specific goal
                        // For 23 clean dots: emphasize contrast and clusters, penalize isolated pixels
                        params.alpha = 2.0f;  // Weight for contrast (INCREASED - prioritize high contrast dots)
                        params.beta = 0.5f;   // Weight for noise (reduce background noise)
                        params.gamma = 2.0f;  // Weight for isolated pixels (heavily penalize single-pixel noise)

                        // Event count constraint - prevent "turn everything off" solutions
                        params.minimum_event_pixels = 500;  // 23 dots × ~25-50 pixels each
                        params.delta = 5.0f;  // Heavy penalty for insufficient events

                        // Get bias ranges from BiasManager
                        EventCameraGeneticOptimizer::Genome::BiasRanges hw_ranges;
                        const auto& bias_ranges = bias_manager.get_bias_ranges();
                        if (!bias_ranges.empty()) {
                            if (bias_ranges.count("bias_diff")) {
                                hw_ranges.diff_min = bias_ranges.at("bias_diff").min;
                                hw_ranges.diff_max = bias_ranges.at("bias_diff").max;
                            }
                            if (bias_ranges.count("bias_refr")) {
                                hw_ranges.refr_min = bias_ranges.at("bias_refr").min;
                                hw_ranges.refr_max = bias_ranges.at("bias_refr").max;
                            }
                            if (bias_ranges.count("bias_fo")) {
                                hw_ranges.fo_min = bias_ranges.at("bias_fo").min;
                                hw_ranges.fo_max = bias_ranges.at("bias_fo").max;
                            }
                            if (bias_ranges.count("bias_hpf")) {
                                hw_ranges.hpf_min = bias_ranges.at("bias_hpf").min;
                                hw_ranges.hpf_max = bias_ranges.at("bias_hpf").max;
                            }
                        }

                        // Fitness callback that applies hw_ranges to each genome
                        auto fitness_callback = [&config, &ga_cfg, hw_ranges](const EventCameraGeneticOptimizer::Genome& genome) {
                            // Create genome copy with hardware ranges applied
                            EventCameraGeneticOptimizer::Genome genome_copy = genome;
                            genome_copy.set_ranges(hw_ranges);
                            genome_copy.clamp();  // Ensure values are within hardware limits
                            return evaluate_genome_fitness(genome_copy, config, ga_cfg.frames_per_eval);
                        };

                        // Progress callback to update UI
                        auto progress_callback = [](int generation, float best_fitness,
                                                   const EventCameraGeneticOptimizer::Genome& best_genome,
                                                   const EventCameraGeneticOptimizer::FitnessResult& best_result) {
                            ga_state.current_generation.store(generation);
                            ga_state.best_fitness.store(best_fitness);
                            {
                                std::lock_guard<std::mutex> lock(ga_state.mutex);
                                ga_state.best_genome = best_genome;
                                ga_state.best_result = best_result;
                            }
                        };

                        // Create optimizer with hardware-aware fitness callback
                        ga_state.optimizer = std::make_unique<EventCameraGeneticOptimizer>(
                            params, fitness_callback, progress_callback);

                        // Launch optimization in separate thread
                        ga_state.optimizer_thread = std::make_unique<std::thread>([&]() {
                            std::cout << "GA optimization thread started" << std::endl;
                            EventCameraGeneticOptimizer::Genome best = ga_state.optimizer->optimize();

                            // Store final results
                            {
                                std::lock_guard<std::mutex> lock(ga_state.mutex);
                                ga_state.best_genome = best;
                                ga_state.best_fitness = ga_state.optimizer->get_best_fitness();
                            }

                            ga_state.running = false;
                            std::cout << "GA optimization thread finished" << std::endl;
                        });

                        std::cout << "GA optimization started in background" << std::endl;
                    }
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Connect camera to start optimization");
                }
            } else {
                // Show progress (while GA is running)
                ImGui::Text("Optimization Running...");

                // Calculate progress
                int current_gen = ga_state.current_generation.load();
                float progress = static_cast<float>(current_gen) / static_cast<float>(ga_cfg.num_generations);
                ImGui::ProgressBar(progress, ImVec2(-1, 0));

                ImGui::Text("Generation: %d / %d", current_gen, ga_cfg.num_generations);
                ImGui::Text("Best Fitness: %.4f", ga_state.best_fitness.load());

                if (ImGui::Button("Stop Optimization", ImVec2(-1, 0))) {
                    std::cout << "Stopping GA optimization..." << std::endl;
                    if (ga_state.optimizer) {
                        ga_state.optimizer->stop();
                    }
                    ga_state.running = false;
                }
            }

            ImGui::Separator();

            // Display best results
            if (ga_state.best_fitness < 1e9f) {
                ImGui::Text("Best Results");
                ImGui::Text("Fitness: %.4f", ga_state.best_fitness.load());
                ImGui::Text("Contrast: %.2f", ga_state.best_result.contrast_score);
                ImGui::Text("Noise: %.4f", ga_state.best_result.noise_metric);

                if (ImGui::CollapsingHeader("Best Parameters")) {
                    ImGui::Text("Biases:");
                    ImGui::Text("  diff=%d refr=%d fo=%d hpf=%d",
                               ga_state.best_genome.bias_diff,
                               ga_state.best_genome.bias_refr,
                               ga_state.best_genome.bias_fo,
                               ga_state.best_genome.bias_hpf);
                    ImGui::Text("Accumulation: %.3f s", ga_state.best_genome.accumulation_time_s);
                }

                if (!ga_state.running && app_state->camera_state().is_connected()) {
                    if (ImGui::Button("Apply Best Parameters", ImVec2(-1, 0))) {
                        // Clamp genome to hardware ranges before applying
                        EventCameraGeneticOptimizer::Genome clamped_genome = ga_state.best_genome;

                        // Get hardware ranges from BiasManager and clamp
                        EventCameraGeneticOptimizer::Genome::BiasRanges hw_ranges;
                        const auto& bias_ranges = bias_manager.get_bias_ranges();
                        if (!bias_ranges.empty()) {
                            if (bias_ranges.count("bias_diff")) {
                                hw_ranges.diff_min = bias_ranges.at("bias_diff").min;
                                hw_ranges.diff_max = bias_ranges.at("bias_diff").max;
                            }
                            if (bias_ranges.count("bias_refr")) {
                                hw_ranges.refr_min = bias_ranges.at("bias_refr").min;
                                hw_ranges.refr_max = bias_ranges.at("bias_refr").max;
                            }
                            if (bias_ranges.count("bias_fo")) {
                                hw_ranges.fo_min = bias_ranges.at("bias_fo").min;
                                hw_ranges.fo_max = bias_ranges.at("bias_fo").max;
                            }
                            if (bias_ranges.count("bias_hpf")) {
                                hw_ranges.hpf_min = bias_ranges.at("bias_hpf").min;
                                hw_ranges.hpf_max = bias_ranges.at("bias_hpf").max;
                            }
                        }
                        clamped_genome.set_ranges(hw_ranges);
                        clamped_genome.clamp();

                        // Apply clamped values to config and camera
                        config.camera_settings().bias_diff = clamped_genome.bias_diff;
                        config.camera_settings().bias_refr = clamped_genome.bias_refr;
                        config.camera_settings().bias_fo = clamped_genome.bias_fo;
                        config.camera_settings().bias_hpf = clamped_genome.bias_hpf;
                        config.camera_settings().accumulation_time_s = clamped_genome.accumulation_time_s;

                        auto& cam_info = app_state->camera_state().camera_manager()->get_camera(0);
                        apply_bias_settings(*cam_info.camera, config.camera_settings());

                        std::cout << "Applied best GA parameters to camera (clamped to hardware limits)" << std::endl;
                    }
                }
            } else {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No optimization results yet");
            }
        }
        // ImGui::End(); // TEMP: Commented out - Advanced window disabled

        // Camera view window
        // Size window to match camera aspect ratio
        float camera_aspect = static_cast<float>(app_state->display_settings().get_image_width()) / app_state->display_settings().get_image_height();
        float feed_window_width = window_width - 450;
        float feed_window_height = (feed_window_width / camera_aspect) + 50;  // +50 for window chrome

        ImGui::SetNextWindowPos(ImVec2(440, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(feed_window_width, feed_window_height),
                                ImGuiCond_FirstUseEver);

        if (ImGui::Begin("Camera Feed")) {
            upload_frame_to_gpu();

            if (app_state->texture_manager().get_texture_id() != 0) {
                ImVec2 window_size = ImGui::GetContentRegionAvail();

                // Maintain aspect ratio
                float aspect = static_cast<float>(app_state->texture_manager().get_width()) / app_state->texture_manager().get_height();
                float display_width = window_size.x;
                float display_height = display_width / aspect;

                if (display_height > window_size.y) {
                    display_height = window_size.y;
                    display_width = display_height * aspect;
                }

                ImGui::Image((void*)(intptr_t)app_state->texture_manager().get_texture_id(),
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
    app_state->request_shutdown();

    // Stop camera if connected
    if (app_state->camera_state().is_connected() && app_state->camera_state().camera_manager()) {
        auto& cam_info = app_state->camera_state().camera_manager()->get_camera(0);
        if (cam_info.camera && cam_info.camera->is_running()) {
            cam_info.camera->stop();
        }
    }

    // Wait for event thread
    if (app_state->camera_state().event_thread() && app_state->camera_state().event_thread()->joinable()) {
        app_state->camera_state().event_thread()->join();
    }

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Cleanup OpenGL and video module (handled by AppState destructor)
    // AppState manages texture_manager, frame_processor, frame_buffer lifecycle

    // Cleanup GLFW
    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "Shutdown complete" << std::endl;
    return 0;
}
