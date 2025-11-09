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

// Local headers
#include "camera_manager.h"
#include "video/simd_utils.h"  // SIMD-accelerated pixel operations
#include "video/gpu_compute.h"  // GPU compute shaders for parallel processing
#include "app_config.h"
#include "event_camera_genetic_optimizer.h"

// Camera features
#include "camera/features/erc_feature.h"
#include "camera/features/antiflicker_feature.h"
#include "camera/features/trail_filter_feature.h"
#include "camera/features/roi_feature.h"
#include "camera/features/monitoring_feature.h"

// Application state module
#include "core/app_state.h"

// UI modules
#include "ui/settings_panel.h"
#include "camera/bias_manager.h"

// Global application state (replaces all previous global state variables)
std::unique_ptr<core::AppState> app_state;

// PERFORMANCE: Removed global framegen_mutex bottleneck!
// Now using per-camera mutexes in CameraState for isolated synchronization.
// Event processing is lock-free (single-threaded per camera).

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

    // Flag to enable GA frame capture
    std::atomic<bool> capturing_for_ga{false};
    video::FrameBuffer ga_frame_buffer;  // Dedicated buffer for GA

    // GPU fitness evaluator for accelerated batch processing (50× faster)
    std::unique_ptr<video::gpu::GPUFitnessEvaluator> gpu_fitness_evaluator;
} ga_state;

// ============================================================================
// Binary Stream Processing Infrastructure
// ============================================================================

/**
 * Create lookup table for BIT extraction (not value ranges)
 * Maps pixels with specific bit set to 255, all others to 0
 */
static cv::Mat create_bit_extraction_lut(int bit_position) {
    cv::Mat lut(1, 256, CV_8U);
    uint8_t mask = (1 << bit_position);
    for (int i = 0; i < 256; i++) {
        lut.at<uchar>(i) = (i & mask) ? 255 : 0;
    }
    return lut;
}

// Dual 1-bit array storage (extracted early in pipeline)
// Storage for 2 cameras, each with 2 bits
struct DualBitStorage {
    cv::Mat bit1;  // First extracted bit (1-bit binary)
    cv::Mat bit2;  // Second extracted bit (1-bit binary)
};
static DualBitStorage dual_bit_arrays[2];  // Index 0 = Camera 0, Index 1 = Camera 1

// Old functions removed - now using dynamic bit extraction with user-selected bit positions

// ============================================================================

/**
 * Store frame in frame buffer
 * Implements frame dropping to prevent queue buildup and maintain real-time display
 */
void update_texture(const cv::Mat& frame, int camera_index) {
    if (frame.empty() || !app_state) return;
    app_state->frame_buffer(camera_index).store_frame(frame);
}

/**
 * EARLY BIT EXTRACTION: Extract both selected bits from incoming frame
 * Works on 1-bit arrays throughout pipeline for maximum efficiency
 * Only converts to 8-bit BGR at the very end for display
 */
void store_bit_extracted_image(const cv::Mat& frame, int camera_index) {
    if (frame.empty() || !app_state) return;

    // === STEP 1: EARLY BIT EXTRACTION (at source) ===
    // Extract grayscale channel (preserves original 8-bit values)
    cv::Mat gray;
    if (frame.channels() == 3) {
        gray = cv::Mat(frame.size(), CV_8UC1);
        cv::extractChannel(frame, gray, 0);  // Use channel 0 to preserve exact values
    } else {
        gray = frame;  // Already single-channel
    }

    // Get user-selected bit positions
    int bit1_pos = static_cast<int>(app_state->display_settings().get_binary_stream_mode());
    int bit2_pos = static_cast<int>(app_state->display_settings().get_binary_stream_mode_2());

    // Extract BOTH bits using LUT (O(1) per pixel)
    cv::Mat lut1 = create_bit_extraction_lut(bit1_pos);
    cv::Mat lut2 = create_bit_extraction_lut(bit2_pos);

    cv::Mat bit1(gray.size(), CV_8UC1);  // 1-bit binary (stored as 0 or 255 in CV_8UC1)
    cv::Mat bit2(gray.size(), CV_8UC1);  // 1-bit binary

    cv::LUT(gray, lut1, bit1);
    cv::LUT(gray, lut2, bit2);

    // Store the extracted 1-bit arrays for this camera
    dual_bit_arrays[camera_index].bit1 = bit1;
    dual_bit_arrays[camera_index].bit2 = bit2;

    // === STEP 2: PROCESS 1-BIT ARRAYS BASED ON DISPLAY MODE ===
    auto display_mode = app_state->display_settings().get_display_mode();
    cv::Mat final_binary;  // Final 1-bit result

    using DisplayMode = core::DisplaySettings::DisplayMode;

    switch (display_mode) {
        case DisplayMode::OR_BEFORE_PROCESSING:
            // OR the bits FIRST, then process the combined 1-bit array
            cv::bitwise_or(bit1, bit2, final_binary);
            // TODO: Add processing pipeline here (GA, filters, etc.)
            break;

        case DisplayMode::OR_AFTER_PROCESSING:
            // Process BOTH bits separately, then OR at the end
            // TODO: Add processing for bit1
            // TODO: Add processing for bit2
            cv::bitwise_or(bit1, bit2, final_binary);
            break;

        case DisplayMode::DISPLAY_BIT_1:
            // Only process and display first bit
            final_binary = bit1;
            // TODO: Add processing pipeline here
            break;

        case DisplayMode::DISPLAY_BIT_2:
            // Only process and display second bit
            final_binary = bit2;
            // TODO: Add processing pipeline here
            break;
    }

    // === STEP 3: CONVERT 1-BIT TO 8-BIT BGR FOR DISPLAY (FINAL STEP) ===
    cv::Mat display_frame;
    cv::cvtColor(final_binary, display_frame, cv::COLOR_GRAY2BGR);

    // Store final BGR frame in buffer for display
    app_state->frame_buffer(camera_index).store_frame(display_frame);
}

// Static storage for last combined frame to avoid flickering
static video::FrameRef last_combined_frame;  // ZERO-COPY: Use FrameRef instead of cv::Mat

/**
 * Combine frames from both cameras by adding pixels (ZERO-COPY optimized)
 * Returns combined frame only when BOTH cameras have frames available
 * Otherwise returns the last successfully combined frame
 */
video::FrameRef combine_camera_frames() {
    if (!app_state) return video::FrameRef();

    // Check if BOTH cameras have unconsumed frames
    bool frame0_ready = app_state->frame_buffer(0).has_unconsumed_frame();
    bool frame1_ready = app_state->frame_buffer(1).has_unconsumed_frame();

    // Only consume and combine if BOTH cameras have frames
    if (!frame0_ready || !frame1_ready) {
        // Return last combined frame to avoid flickering (zero-copy)
        return last_combined_frame;
    }

    // Both frames are ready - consume them (zero-copy FrameRefs)
    auto frame0_opt = app_state->frame_buffer(0).consume_frame();
    auto frame1_opt = app_state->frame_buffer(1).consume_frame();

    // Double-check we got both frames
    if (!frame0_opt || !frame1_opt) {
        return last_combined_frame;
    }

    video::FrameRef frame0_ref = frame0_opt.value();
    video::FrameRef frame1_ref = frame1_opt.value();

    if (frame0_ref.empty() || frame1_ref.empty()) {
        return last_combined_frame;
    }

    // Get writable access for modifications (copy-on-write)
    cv::Mat frame0 = frame0_ref.write();
    cv::Mat frame1 = frame1_ref.write();

    // Flip second view horizontally if requested
    if (app_state->display_settings().get_flip_second_view()) {
        cv::flip(frame1, frame1, 1);  // 1 = flip horizontally
    }

    // Ensure frames are the same size and type
    if (frame0.size() != frame1.size() || frame0.type() != frame1.type()) {
        // Resize frame1 to match frame0 if needed
        cv::resize(frame1, frame1, frame0.size());
        frame1.convertTo(frame1, frame0.type());
    }

    // Add frames together (saturate at 255 for uint8)
    cv::Mat combined;
    cv::add(frame0, frame1, combined);

    // ZERO-COPY: Store as FrameRef (no clone needed!)
    last_combined_frame = video::FrameRef(std::move(combined));

    return last_combined_frame;
}

/**
 * Upload OpenCV frame to OpenGL texture (ZERO-COPY optimized)
 * Binary stream images are already processed and ready for display
 */
void upload_frame_to_gpu(int camera_index) {
    if (!app_state) return;

    // Try to consume a frame from the buffer (zero-copy FrameRef)
    auto frame_opt = app_state->frame_buffer(camera_index).consume_frame();
    if (!frame_opt.has_value()) {
        return;  // No new frame available
    }

    video::FrameRef frame_ref = frame_opt.value();

    // Binary stream images are already fully processed - upload directly
    // No need to apply frame_processor (ROI, subtraction, etc.) again
    video::ReadGuard guard(frame_ref);
    const cv::Mat& frame = guard.get();

    // Upload to GPU texture
    app_state->texture_manager(camera_index).upload_frame(frame);
}

/**
 * Apply camera bias settings
 */
void apply_bias_settings(Metavision::Camera& camera, const AppConfig::CameraSettings& settings) {
    auto* i_ll_biases = camera.get_device().get_facility<Metavision::I_LL_Biases>();
    if (i_ll_biases) {
        std::cout << "Applying camera biases..." << std::endl;

        // Helper to set a bias with error handling
        auto set_bias = [&](const char* name, int value) {
            try {
                i_ll_biases->set(name, value);
                std::cout << "  " << name << "=" << value << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "  Warning: Could not set " << name << ": " << e.what() << std::endl;
            }
        };

        // Apply all biases
        set_bias("bias_diff", settings.bias_diff);
        set_bias("bias_diff_on", settings.bias_diff_on);
        set_bias("bias_diff_off", settings.bias_diff_off);
        set_bias("bias_refr", settings.bias_refr);
        set_bias("bias_fo", settings.bias_fo);
        set_bias("bias_hpf", settings.bias_hpf);

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
                i_ll_biases->set("bias_diff_on", genome.bias_diff_on);
                i_ll_biases->set("bias_diff_off", genome.bias_diff_off);
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

            // PERFORMANCE: Use per-camera mutex (camera 0 for GA)
            std::lock_guard<std::mutex> lock(app_state->camera_state().frame_gen_mutex(0));
            int width = app_state->display_settings().get_image_width();
            int height = app_state->display_settings().get_image_height();
            app_state->camera_state().frame_generator() = std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(
                width, height, accumulation_time_us);
            app_state->camera_state().frame_generator()->set_output_callback([](const Metavision::timestamp ts, cv::Mat& frame) {
                if (frame.empty() || !app_state) return;

                // ZERO-COPY: Create FrameRef from frame (shares data)
                video::FrameRef frame_ref(frame);

                // Store RAW frame for GA if capturing (before any processing)
                if (ga_state.capturing_for_ga) {
                    ga_state.ga_frame_buffer.store_frame(frame_ref);  // Zero-copy share
                }

                // Display processing now handled by store_bit_extracted_image()
                // which does early bit extraction and works on 1-bit arrays
                cv::Mat display_frame = frame_ref.write();

                // Rate limit display updates to target FPS (GA uses camera 0)
                auto now = std::chrono::steady_clock::now();
                auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
                int fps_target = app_state->display_settings().get_target_fps();

                // Check if enough time has passed since last display for camera 0
                if (app_state->frame_sync(0).should_display_frame(now_us, fps_target)) {
                    app_state->frame_sync(0).on_frame_generated(ts, now_us);
                    app_state->frame_sync(0).on_frame_displayed(now_us);
                    update_texture(display_frame, 0);  // GA test uses camera 0
                }
            });
        }

        // Apply other genome parameters (trail filter, antiflicker, ERC, etc.)
        if (genome.enable_trail_filter) {
            auto* trail_filter = cam_info.camera->get_device().get_facility<Metavision::I_EventTrailFilterModule>();
            if (trail_filter) {
                trail_filter->set_type(Metavision::I_EventTrailFilterModule::Type::STC_KEEP_TRAIL);
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
    std::this_thread::sleep_for(std::chrono::milliseconds(config.runtime_settings().ga_parameter_settle_ms));

    // Enable GA frame capture
    ga_state.capturing_for_ga = true;

    // Capture frames using dedicated GA buffer
    std::vector<cv::Mat> captured_frames;
    captured_frames.reserve(num_frames);

    int max_attempts = num_frames * config.runtime_settings().ga_frame_capture_max_attempts;
    int attempts = 0;

    while (captured_frames.size() < static_cast<size_t>(num_frames) && attempts < max_attempts) {
        auto frame_opt = ga_state.ga_frame_buffer.consume_frame();
        if (frame_opt.has_value() && !frame_opt.value().empty()) {
            // Extract cv::Mat from FrameRef (need to clone for GA processing)
            video::ReadGuard guard(frame_opt.value());
            captured_frames.push_back(guard.get().clone());
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(config.runtime_settings().ga_frame_capture_wait_ms));
        attempts++;
    }

    // Disable GA frame capture
    ga_state.capturing_for_ga = false;

    // Calculate fitness metrics from captured frames
    if (captured_frames.empty()) {
        std::cerr << "GA ERROR: No frames captured after " << attempts << " attempts. "
                  << "Check camera streaming and frame generation!" << std::endl;
        result.combined_fitness = 1e9f;  // Very bad fitness
        return result;
    }

    if (captured_frames.size() < static_cast<size_t>(num_frames)) {
        std::cerr << "GA WARNING: Only captured " << captured_frames.size()
                  << "/" << num_frames << " frames" << std::endl;
    }

    result.num_valid_frames = captured_frames.size();
    result.total_frames = num_frames;

    // Convert all GA frames to grayscale for GPU processing
    // (Binary stream processing now done early in store_bit_extracted_image())
    std::vector<cv::Mat> grayscale_frames;
    grayscale_frames.reserve(captured_frames.size());

    for (auto& frame : captured_frames) {
        // Convert to grayscale for GPU processing
        cv::Mat gray;
        if (frame.channels() == 3) {
            gray = cv::Mat(frame.size(), CV_8UC1);
            video::simd::bgr_to_gray(frame, gray);
        } else {
            gray = frame.clone();
        }
        grayscale_frames.push_back(gray);

        // Ensure BGR for downstream processing
        if (frame.channels() == 1) {
            cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
        }
    }

    // GPU-accelerated batch fitness evaluation (50× faster)
    // Evaluate all frames in parallel on GPU for quick metrics
    std::vector<float> gpu_metrics;
    if (ga_state.gpu_fitness_evaluator && !grayscale_frames.empty()) {
        ga_state.gpu_fitness_evaluator->evaluate_batch(grayscale_frames, gpu_metrics);

        // Use GPU metrics for basic fitness scoring
        if (!gpu_metrics.empty()) {
            result.mean_brightness = gpu_metrics[0];  // GPU provides more accurate aggregated metrics
            std::cout << "GPU fitness eval: " << gpu_metrics.size() << " frames, "
                     << "mean=" << result.mean_brightness << std::endl;
        }
    }

    // If cluster filter is enabled, use cluster-based metrics
    if (config.ga_settings().enable_cluster_filter && !config.ga_settings().cluster_centers.empty()) {
        // Calculate cluster-based metrics: reward clustered pixels, penalize isolated noise (SIMD-accelerated)
        cv::Mat gray;
        if (captured_frames[0].channels() == 3) {
            gray = cv::Mat(captured_frames[0].size(), CV_8UC1);
            video::simd::bgr_to_gray(captured_frames[0], gray);  // 7.5× faster
        } else {
            gray = captured_frames[0];
        }

        // Threshold to get binary events
        cv::Mat binary;
        cv::threshold(gray, binary, 10, 255, cv::THRESH_BINARY);

        int total_pixels = cv::countNonZero(binary);
        if (total_pixels == 0) {
            result.contrast_score = 0.01f;
            result.noise_metric = 0.0f;
            result.isolated_pixel_ratio = 0.0f;
            result.cluster_fill_metric = 1.0f;
        } else {
            // Create mask for cluster regions (signal areas)
            cv::Mat cluster_mask = cv::Mat::zeros(gray.size(), CV_8U);
            int cluster_radius = config.ga_settings().cluster_radius;
            for (const auto& center : config.ga_settings().cluster_centers) {
                cv::Point pt(center.first, center.second);
                cv::circle(cluster_mask, pt, cluster_radius, cv::Scalar(255), -1);
            }

            // Count pixels in cluster regions (signal) vs outside (noise)
            cv::Mat pixels_in_clusters, pixels_outside_clusters;
            cv::bitwise_and(binary, cluster_mask, pixels_in_clusters);
            cv::bitwise_and(binary, ~cluster_mask, pixels_outside_clusters);

            int signal_pixels = cv::countNonZero(pixels_in_clusters);
            int noise_pixels = cv::countNonZero(pixels_outside_clusters);

            // Contrast score: reward high signal in clusters
            // Use connected component analysis for cluster quality
            float component_fitness = EventCameraGeneticOptimizer::calculate_connected_component_fitness(
                captured_frames[0],
                config.ga_settings().cluster_radius,
                config.ga_settings().min_cluster_radius);

            result.contrast_score = std::max(0.1f, static_cast<float>(signal_pixels) / 10.0f);

            // Noise metric: penalize ONLY pixels OUTSIDE cluster regions
            // Higher noise_pixels = worse
            result.noise_metric = static_cast<float>(noise_pixels) / 100.0f;

            // Calculate fill metric for cluster quality
            result.cluster_fill_metric = EventCameraGeneticOptimizer::calculate_cluster_fill(
                captured_frames[0], config.ga_settings().min_cluster_radius);

            // Isolated pixel ratio based on noise outside clusters
            result.isolated_pixel_ratio = total_pixels > 0 ?
                static_cast<float>(noise_pixels) / static_cast<float>(total_pixels) : 0.0f;
        }

    } else {
        // Standard contrast/noise calculation
        float total_contrast = 0.0f;
        for (const auto& frame : captured_frames) {
            total_contrast += EventCameraGeneticOptimizer::calculate_contrast(frame);
        }
        result.contrast_score = total_contrast / captured_frames.size();

        // Calculate noise metrics
        result.temporal_variance = EventCameraGeneticOptimizer::calculate_temporal_variance(captured_frames);
        result.spatial_noise = EventCameraGeneticOptimizer::calculate_spatial_noise(captured_frames[0]);
        result.noise_metric = result.temporal_variance + result.spatial_noise;

        // Calculate isolated pixel ratio (single-pixel noise) with radius 2
        result.isolated_pixel_ratio = EventCameraGeneticOptimizer::calculate_isolated_pixels(
            captured_frames[0], 2);

        // Calculate fill metric for cluster quality
        result.cluster_fill_metric = EventCameraGeneticOptimizer::calculate_cluster_fill(
            captured_frames[0], 2);
    }

    // Calculate total event pixels (bright pixels above threshold) - SIMD-accelerated
    cv::Mat gray;
    if (captured_frames[0].channels() == 3) {
        gray = cv::Mat(captured_frames[0].size(), CV_8UC1);
        video::simd::bgr_to_gray(captured_frames[0], gray);  // 7.5× faster
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
                        const std::string& serial_hint = "", bool start_camera = true) {
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
    if (app_state->camera_state().is_simulation_mode() && app_state->camera_state().event_thread(0)) {
        app_state->request_shutdown();
        if (app_state->camera_state().event_thread(0)->joinable()) {
            app_state->camera_state().event_thread(0)->join();
        }
        // Reset running state
        app_state = std::make_unique<core::AppState>();
    }

    // Initialize camera manager - auto-detect all available cameras
    app_state->camera_state().camera_manager() = std::make_unique<CameraManager>();

    // Try to initialize dual cameras (empty strings = auto-detect)
    int num_cameras = app_state->camera_state().camera_manager()->initialize("", "");

    if (num_cameras == 0) {
        std::cerr << "Failed to initialize any cameras" << std::endl;
        app_state->camera_state().camera_manager() = nullptr;
        return false;
    }

    std::cout << "Initialized " << num_cameras << " camera(s)" << std::endl;

    // Use first camera for display settings
    auto& cam_info = app_state->camera_state().camera_manager()->get_camera(0);
    app_state->display_settings().set_image_size(cam_info.width, cam_info.height);

    // Initialize BiasManager with all cameras
    if (bias_mgr.initialize(*cam_info.camera)) {
        std::cout << "BiasManager initialized with Camera 0" << std::endl;

        // Add additional cameras to BiasManager
        for (int i = 1; i < num_cameras; ++i) {
            auto& additional_cam = app_state->camera_state().camera_manager()->get_camera(i);
            if (bias_mgr.add_camera(*additional_cam.camera)) {
                std::cout << "BiasManager: Added Camera " << i << std::endl;
            }
        }
    } else {
        std::cerr << "Warning: Failed to initialize BiasManager" << std::endl;
    }

    // Register and initialize hardware features (using Camera 0 for UI)
    std::cout << "\nRegistering hardware features..." << std::endl;
    app_state->feature_manager().register_feature(std::make_shared<EventCamera::ERCFeature>());
    app_state->feature_manager().register_feature(std::make_shared<EventCamera::AntiFlickerFeature>());
    app_state->feature_manager().register_feature(std::make_shared<EventCamera::TrailFilterFeature>());
    app_state->feature_manager().register_feature(std::make_shared<EventCamera::ROIFeature>(app_state->roi_filter(), app_state->display_settings()));
    app_state->feature_manager().register_feature(std::make_shared<EventCamera::MonitoringFeature>());

    // Initialize features on Camera 0
    app_state->feature_manager().initialize_all(*cam_info.camera);

    // Add additional cameras to FeatureManager
    for (int i = 1; i < num_cameras; ++i) {
        auto& additional_cam = app_state->camera_state().camera_manager()->get_camera(i);
        if (app_state->feature_manager().add_camera(*additional_cam.camera)) {
            std::cout << "FeatureManager: Added Camera " << i << std::endl;
        }
    }
    std::cout << "Hardware features initialized (controlling all cameras)\n" << std::endl;

    // Apply Bias settings from config to ALL cameras
    for (int i = 0; i < num_cameras; ++i) {
        auto& cam = app_state->camera_state().camera_manager()->get_camera(i);
        std::cout << "\nApplying bias settings to Camera " << i << "..." << std::endl;
        apply_bias_settings(*cam.camera, config.camera_settings());
    }

    // Apply Trail Filter settings from config to ALL cameras directly
    Metavision::I_EventTrailFilterModule::Type trail_type;
    switch (config.camera_settings().trail_filter_type) {
        case 0: trail_type = Metavision::I_EventTrailFilterModule::Type::TRAIL; break;
        case 1: trail_type = Metavision::I_EventTrailFilterModule::Type::STC_CUT_TRAIL; break;
        case 2: trail_type = Metavision::I_EventTrailFilterModule::Type::STC_KEEP_TRAIL; break;
        default: trail_type = Metavision::I_EventTrailFilterModule::Type::STC_KEEP_TRAIL; break;
    }

    for (int i = 0; i < num_cameras; ++i) {
        auto& cam = app_state->camera_state().camera_manager()->get_camera(i);
        auto* trail_filter = cam.camera->get_device().get_facility<Metavision::I_EventTrailFilterModule>();

        if (trail_filter) {
            try {
                trail_filter->set_type(trail_type);
                trail_filter->set_threshold(config.camera_settings().trail_filter_threshold);
                trail_filter->enable(config.camera_settings().trail_filter_enabled);
                std::cout << "Trail Filter configured for Camera " << i << " from config: "
                          << (config.camera_settings().trail_filter_enabled ? "enabled" : "disabled")
                          << ", type=" << config.camera_settings().trail_filter_type
                          << ", threshold=" << config.camera_settings().trail_filter_threshold << "μs" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to configure Trail Filter for Camera " << i << ": " << e.what() << std::endl;
            }
        }
    }

    // Sync TrailFilterFeature with camera state after config is applied
    if (auto trail_feature = std::dynamic_pointer_cast<EventCamera::TrailFilterFeature>(
            app_state->feature_manager().get_feature("Trail Filter"))) {
        trail_feature->sync_from_camera();
    }

    // Create frame generators and set up event callbacks for all cameras
    const uint32_t accumulation_time_us = static_cast<uint32_t>(
        config.camera_settings().accumulation_time_s * 1000000);

    // num_cameras already defined above from camera initialization
    for (int i = 0; i < num_cameras; ++i) {
        auto& cam_info = app_state->camera_state().camera_manager()->get_camera(i);

        // Get camera geometry
        const auto& geom = cam_info.camera->geometry();
        uint16_t width = geom.width();
        uint16_t height = geom.height();

        // Create frame generator for this camera
        app_state->camera_state().frame_generator(i) = std::make_unique<Metavision::PeriodicFrameGenerationAlgorithm>(
            width, height, accumulation_time_us);

        // Set up frame generator output callback (must be done before starting camera)
        int camera_index = i;  // Capture by value for lambda
        app_state->camera_state().frame_generator(i)->set_output_callback(
            [camera_index](const Metavision::timestamp ts, cv::Mat& frame) {
                if (frame.empty() || !app_state) return;

                // Rate limit display updates to target FPS (per-camera)
                auto now = std::chrono::steady_clock::now();
                auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
                int fps_target = app_state->display_settings().get_target_fps();

                // Check if enough time has passed since last display for THIS camera
                if (app_state->frame_sync(camera_index).should_display_frame(now_us, fps_target)) {
                    app_state->frame_sync(camera_index).on_frame_generated(ts, now_us);
                    app_state->frame_sync(camera_index).on_frame_displayed(now_us);

                    // Extract selected bit and store for display
                    store_bit_extracted_image(frame, camera_index);
                }
                // Note: frame_buffer tracks its own dropped/generated statistics
            });

        // Set up event callback to feed events to frame generator
        int64_t max_event_age_us = config.runtime_settings().max_event_age_us;  // Capture for lambda
        cam_info.camera->cd().add_callback(
            [camera_index, max_event_age_us](const Metavision::EventCD* begin, const Metavision::EventCD* end) {
                if (begin == end || !app_state) return;

                // Get current time
                auto now = std::chrono::steady_clock::now();
                auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

                // Count events for event rate calculation
                int64_t event_count = end - begin;

                // Track last event timestamp to measure event latency
                Metavision::timestamp last_ts = (end-1)->t;

                // Record events for metrics
                app_state->event_metrics().record_events(event_count, now_us);
                app_state->event_metrics().record_event_timestamp(last_ts);

                // CRITICAL: Skip old event batches to prevent latency buildup
                // Check age of newest event - if old, skip entire batch
                int64_t cam_start = app_state->camera_state().get_camera_start_time_us();
                Metavision::timestamp newest_event_ts = (end-1)->t;
                int64_t newest_event_system_ts = cam_start + newest_event_ts;
                int64_t event_age_us = now_us - newest_event_system_ts;

                // Skip batches with events older than threshold to stay current
                if (event_age_us > max_event_age_us) {
                    // Skip old data - don't process into frame generator
                    return;
                }

                // PERFORMANCE: LOCK-FREE EVENT PROCESSING!
                // Event callbacks are single-threaded per camera, so no mutex needed.
                // Each camera has isolated frame generator - zero contention!
                if (app_state && app_state->camera_state().frame_generator(camera_index)) {
                    app_state->camera_state().frame_generator(camera_index)->process_events(begin, end);
                }
            });

        std::cout << "Frame generator and event callback configured for camera " << i
                  << " (" << cam_info.serial << ")" << std::endl;
    }

    if (!start_camera) {
        // Cameras initialized but not started - caller will start them later
        app_state->camera_state().set_connected(true);
        app_state->camera_state().set_simulation_mode(false);
        std::cout << "Cameras connected but not started (will start after UI initialization)" << std::endl;
        return true;
    }

    // Start all cameras
    auto start_time = std::chrono::steady_clock::now();
    int64_t start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(start_time.time_since_epoch()).count();
    app_state->camera_state().set_camera_start_time_us(start_time_us);

    for (int i = 0; i < num_cameras; ++i) {
        auto& cam_info = app_state->camera_state().camera_manager()->get_camera(i);
        cam_info.camera->start();
        std::cout << "Camera " << i << " started: " << cam_info.serial << std::endl;
    }

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

    // Initialize BiasManager for camera bias control
    EventCamera::BiasManager bias_manager;

    // Camera/Simulation Mode
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

        // Use try_connect_camera to properly initialize everything including features
        // Don't start camera yet - wait until after UI is initialized
        if (!try_connect_camera(config, bias_manager, available_cameras[0], false)) {
            std::cerr << "ERROR: Failed to connect to camera - starting in SIMULATION MODE" << std::endl;
            app_state->camera_state().set_simulation_mode(true);
            app_state->display_settings().set_image_size(1280, 720);
        } else {
            std::cout << "Successfully connected to camera (not started yet)" << std::endl;
        }
    }

    std::cout << "Debug: Camera initialization complete!" << std::endl;

    // NOTE: Frame generation and callbacks are set up by try_connect_camera()

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

    // Initialize GPU compute infrastructure for GA fitness evaluation
    std::cout << "Initializing GPU compute shaders..." << std::endl;
    ga_state.gpu_fitness_evaluator = std::make_unique<video::gpu::GPUFitnessEvaluator>();

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
        std::cout << "\nStarting cameras..." << std::endl;

        // Record camera start time for event latency calculation
        auto start_time = std::chrono::steady_clock::now();
        int64_t start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(start_time.time_since_epoch()).count();
        app_state->camera_state().set_camera_start_time_us(start_time_us);

        // Start all cameras (event callbacks already set up by try_connect_camera)
        int num_cameras = app_state->camera_state().num_cameras();
        for (int i = 0; i < num_cameras; ++i) {
            auto& cam_info = app_state->camera_state().camera_manager()->get_camera(i);
            cam_info.camera->start();
            std::cout << "Camera " << i << " started: " << cam_info.serial << std::endl;
        }

        std::cout << "\nPress ESC or close window to exit\n" << std::endl;
    } else {
        // Simulation mode - generate synthetic frames
        std::cout << "\nStarting SIMULATION mode..." << std::endl;
        std::cout << "Generating synthetic event camera frames" << std::endl;
        std::cout << "\nPress ESC or close window to exit\n" << std::endl;

        app_state->camera_state().event_thread(0) = std::make_unique<std::thread>([&]() {
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
                update_texture(sim_frame, 0);  // Simulation uses camera 0

                // Sleep to simulate frame rate
                std::this_thread::sleep_for(std::chrono::milliseconds(config.runtime_settings().simulation_frame_delay_ms));
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
                // Get last frame (zero-copy FrameRef)
                video::FrameRef frame_ref = app_state->texture_manager().get_last_frame();
                if (!frame_ref.empty()) {
                    // Extract cv::Mat for ImageJ streaming
                    video::ReadGuard guard(frame_ref);
                    cv::Mat frame = guard.get().clone();
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

        // Render settings panel (hidden - we'll render sections manually below)
        // settings_panel.render();

        // Handle camera connection requests from SettingsPanel
        if (settings_panel.camera_reconnect_requested()) {
            settings_panel.reset_camera_reconnect_request();

            std::cout << "\n=== Disconnecting cameras ===" << std::endl;

            // Stop all cameras
            if (app_state->camera_state().is_connected() && app_state->camera_state().camera_manager()) {
                int num_cameras = app_state->camera_state().num_cameras();
                for (int i = 0; i < num_cameras; ++i) {
                    try {
                        auto& cam_info = app_state->camera_state().camera_manager()->get_camera(i);
                        cam_info.camera->stop();
                        std::cout << "Camera " << i << " stopped" << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Error stopping camera " << i << ": " << e.what() << std::endl;
                    }
                }
            }

            // Reset the running flag so camera can start again
            app_state->reset_running_flag();

            // Shutdown and clear hardware features
            app_state->feature_manager().shutdown_all();
            app_state->feature_manager().clear();

            // Clear camera resources (frame generators, texture managers, and camera manager)
            for (int i = 0; i < 2; ++i) {  // MAX_CAMERAS = 2 (one per physical camera)
                app_state->camera_state().frame_generator(i).reset();
                app_state->texture_manager(i).reset();
            }
            app_state->camera_state().camera_manager().reset();
            app_state->camera_state().set_connected(false);

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

        // GA optimization UI is now in the "Genetic Optimization" strip
        // (Removed duplicate GA UI section that was here)

        if (false) {  // Disabled - GA UI moved to settings strips
            auto& ga_cfg = config.ga_settings();

            if (!ga_state.running) {
                ImGui::Text("Optimization Parameters");
                ImGui::Separator();

                // Row 1: Population Size | Generations
                ImGui::PushItemWidth(100);
                if (ImGui::InputInt("Population Size", &ga_cfg.population_size)) {
                    ga_cfg.population_size = std::max(4, ga_cfg.population_size);
                }
                ImGui::SameLine();
                if (ImGui::InputInt("Generations", &ga_cfg.num_generations)) {
                    ga_cfg.num_generations = std::max(1, ga_cfg.num_generations);
                }
                ImGui::PopItemWidth();

                // Row 2: Mutation Rate | Crossover Rate
                ImGui::PushItemWidth(100);
                if (ImGui::InputFloat("Mutation Rate", &ga_cfg.mutation_rate, 0.01f, 0.1f, "%.2f")) {
                    ga_cfg.mutation_rate = std::clamp(ga_cfg.mutation_rate, 0.0f, 1.0f);
                }
                ImGui::SameLine();
                if (ImGui::InputFloat("Crossover Rate", &ga_cfg.crossover_rate, 0.01f, 0.1f, "%.2f")) {
                    ga_cfg.crossover_rate = std::clamp(ga_cfg.crossover_rate, 0.0f, 1.0f);
                }
                ImGui::PopItemWidth();

                // Row 3: Frames per Evaluation
                ImGui::PushItemWidth(100);
                if (ImGui::InputInt("Frames/Eval", &ga_cfg.frames_per_eval)) {
                    ga_cfg.frames_per_eval = std::max(1, ga_cfg.frames_per_eval);
                }
                ImGui::PopItemWidth();

                ImGui::Separator();
                ImGui::Text("Parameters to Optimize:");

                ImGui::Checkbox("bias_diff##ga_opt", &ga_cfg.optimize_bias_diff); ImGui::SameLine();
                ImGui::Checkbox("bias_diff_on##ga_opt", &ga_cfg.optimize_bias_diff_on);
                ImGui::Checkbox("bias_diff_off##ga_opt", &ga_cfg.optimize_bias_diff_off); ImGui::SameLine();
                ImGui::Checkbox("bias_refr##ga_opt", &ga_cfg.optimize_bias_refr);
                ImGui::Checkbox("bias_fo##ga_opt", &ga_cfg.optimize_bias_fo); ImGui::SameLine();
                ImGui::Checkbox("bias_hpf##ga_opt", &ga_cfg.optimize_bias_hpf);
                ImGui::Checkbox("accumulation##ga_opt", &ga_cfg.optimize_accumulation);
                ImGui::Checkbox("trail_filter##ga_opt", &ga_cfg.optimize_trail_filter); ImGui::SameLine();
                ImGui::Checkbox("antiflicker##ga_opt", &ga_cfg.optimize_antiflicker);
                ImGui::Checkbox("erc##ga_opt", &ga_cfg.optimize_erc);

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

                        // Set optimization mask from config checkboxes
                        params.opt_mask.bias_diff = ga_cfg.optimize_bias_diff;
                        params.opt_mask.bias_diff_on = ga_cfg.optimize_bias_diff_on;
                        params.opt_mask.bias_diff_off = ga_cfg.optimize_bias_diff_off;
                        params.opt_mask.bias_refr = ga_cfg.optimize_bias_refr;
                        params.opt_mask.bias_fo = false;  // Currently no checkbox for these
                        params.opt_mask.bias_hpf = false;
                        params.opt_mask.accumulation = ga_cfg.optimize_accumulation;
                        params.opt_mask.trail_filter = ga_cfg.optimize_trail_filter;
                        params.opt_mask.antiflicker = false;  // No checkbox for these
                        params.opt_mask.erc = false;

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
                    // Wait for thread to finish before allowing restart
                    if (ga_state.optimizer_thread && ga_state.optimizer_thread->joinable()) {
                        ga_state.optimizer_thread->join();
                    }
                    ga_state.running = false;
                    std::cout << "GA optimization stopped successfully" << std::endl;
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
                        config.camera_settings().bias_diff_on = clamped_genome.bias_diff_on;
                        config.camera_settings().bias_diff_off = clamped_genome.bias_diff_off;
                        config.camera_settings().bias_refr = clamped_genome.bias_refr;
                        config.camera_settings().bias_fo = clamped_genome.bias_fo;
                        config.camera_settings().bias_hpf = clamped_genome.bias_hpf;
                        config.camera_settings().accumulation_time_s = clamped_genome.accumulation_time_s;

                        // Apply to ALL cameras
                        int num_cameras = app_state->camera_state().num_cameras();
                        for (int i = 0; i < num_cameras; ++i) {
                            auto& cam_info = app_state->camera_state().camera_manager()->get_camera(i);
                            apply_bias_settings(*cam_info.camera, config.camera_settings());
                        }

                        std::cout << "Applied best GA parameters to all cameras (clamped to hardware limits)" << std::endl;
                    }
                }
            } else {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No optimization results yet");
            }
        }
        // ImGui::End(); // TEMP: Commented out - Advanced window disabled

        // === NEW LAYOUT: Cameras side-by-side at top, settings strips below ===
        int num_cameras = app_state->camera_state().num_cameras();
        float camera_aspect = static_cast<float>(app_state->display_settings().get_image_width()) / app_state->display_settings().get_image_height();

        // Calculate layout dimensions
        float cam_spacing = 10.0f;
        float single_cam_width = (window_width - 3 * cam_spacing) / 2.0f;  // Split width in half
        float single_cam_height = (single_cam_width / camera_aspect) + 30.0f;  // +30 for title bar
        float settings_top = single_cam_height + 20.0f;  // Settings start below cameras

        // Check if we should display combined view or separate views
        bool add_images_mode = app_state->display_settings().get_add_images_mode();

        if (add_images_mode && num_cameras >= 2) {
            // === COMBINED VIEW MODE: Single window showing added frames ===
            float combined_width = window_width - 2 * cam_spacing;
            float combined_height = (combined_width / camera_aspect) + 30.0f;

            ImGui::SetNextWindowPos(ImVec2(cam_spacing, cam_spacing), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(combined_width, combined_height), ImGuiCond_FirstUseEver);

            if (ImGui::Begin("Combined Camera View (Added)")) {
                // Combine frames from both cameras (ZERO-COPY)
                video::FrameRef combined_frame_ref = combine_camera_frames();

                if (!combined_frame_ref.empty()) {
                    // Process combined frame
                    video::ReadGuard guard(combined_frame_ref);
                    cv::Mat processed_frame = app_state->frame_processor().process(guard.get());

                    // Upload to texture manager 0
                    app_state->texture_manager(0).upload_frame(processed_frame);
                }

                if (app_state->texture_manager(0).get_texture_id() != 0) {
                    ImVec2 window_size = ImGui::GetContentRegionAvail();

                    // Maintain aspect ratio
                    float aspect = static_cast<float>(app_state->texture_manager(0).get_width()) / app_state->texture_manager(0).get_height();
                    float display_width = window_size.x;
                    float display_height = display_width / aspect;

                    if (display_height > window_size.y) {
                        display_height = window_size.y;
                        display_width = display_height * aspect;
                    }

                    ImGui::Image((void*)(intptr_t)app_state->texture_manager(0).get_texture_id(),
                               ImVec2(display_width, display_height));
                } else {
                    ImGui::Text("Waiting for camera frames...");
                }
            }
            ImGui::End();

            // Update settings_top for combined view
            settings_top = combined_height + 20.0f;

        } else {
            // === SEPARATE VIEW MODE: Two camera windows side-by-side ===
            // Camera 0 (Left) | Camera 1 (Right)
            // Bit selection from dropdown applies to both cameras

            // Camera 0 (Left)
            if (num_cameras > 0) {
                ImGui::SetNextWindowPos(ImVec2(cam_spacing, cam_spacing), ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowSize(ImVec2(single_cam_width, single_cam_height), ImGuiCond_FirstUseEver);

                if (ImGui::Begin("Camera 0")) {
                    upload_frame_to_gpu(0);  // Index 0 = Camera 0

                    if (app_state->texture_manager(0).get_texture_id() != 0) {
                        ImVec2 window_size = ImGui::GetContentRegionAvail();

                        // Maintain aspect ratio
                        float aspect = static_cast<float>(app_state->texture_manager(0).get_width()) / app_state->texture_manager(0).get_height();
                        float display_width = window_size.x;
                        float display_height = display_width / aspect;

                        if (display_height > window_size.y) {
                            display_height = window_size.y;
                            display_width = display_height * aspect;
                        }

                        ImGui::Image((void*)(intptr_t)app_state->texture_manager(0).get_texture_id(),
                                   ImVec2(display_width, display_height));
                    } else {
                        ImGui::Text("Waiting for camera 0 frames...");
                    }
                }
                ImGui::End();
            }

            // Camera 1 (Right)
            if (num_cameras > 1) {
                ImGui::SetNextWindowPos(ImVec2(single_cam_width + 2 * cam_spacing, cam_spacing), ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowSize(ImVec2(single_cam_width, single_cam_height), ImGuiCond_FirstUseEver);

                if (ImGui::Begin("Camera 1")) {
                    upload_frame_to_gpu(1);  // Index 1 = Camera 1

                    if (app_state->texture_manager(1).get_texture_id() != 0) {
                        ImVec2 window_size = ImGui::GetContentRegionAvail();

                        // Maintain aspect ratio
                        float aspect = static_cast<float>(app_state->texture_manager(1).get_width()) / app_state->texture_manager(1).get_height();
                        float display_width = window_size.x;
                        float display_height = display_width / aspect;

                        if (display_height > window_size.y) {
                            display_height = window_size.y;
                            display_width = display_height * aspect;
                        }

                        ImGui::Image((void*)(intptr_t)app_state->texture_manager(1).get_texture_id(),
                                   ImVec2(display_width, display_height));
                    } else {
                        ImGui::Text("Waiting for camera 1 frames...");
                    }
                }
                ImGui::End();
            }

            // Update settings_top to account for 1 row of cameras
            settings_top = single_cam_height + 20.0f;
        }

        // === Settings strips below cameras ===
        float settings_height = window_height - settings_top - 10.0f;
        float strip_width = (window_width - 6 * cam_spacing) / 5.0f;  // 5 strips

        // Strip 1: Analog Biases
        ImGui::SetNextWindowPos(ImVec2(cam_spacing, settings_top), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(strip_width, settings_height), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Analog Biases")) {
            settings_panel.render_bias_controls();
        }
        ImGui::End();

        // Strip 2: Digital Filters
        ImGui::SetNextWindowPos(ImVec2(strip_width + 2 * cam_spacing, settings_top), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(strip_width, settings_height), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Digital Filters")) {
            settings_panel.render_digital_features();

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            settings_panel.render_frame_generation();
        }
        ImGui::End();

        // Strip 3: Genetic Optimization
        ImGui::SetNextWindowPos(ImVec2(2 * strip_width + 3 * cam_spacing, settings_top), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(strip_width, settings_height), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Genetic Optimization")) {

            auto& ga_cfg = config.ga_settings();

            if (!ga_state.running) {
                ImGui::Text("Optimization Parameters");
                ImGui::Separator();

                // Row 1: Population Size | Generations
                ImGui::PushItemWidth(100);
                if (ImGui::InputInt("Population Size", &ga_cfg.population_size)) {
                    ga_cfg.population_size = std::max(4, ga_cfg.population_size);
                }
                ImGui::SameLine();
                if (ImGui::InputInt("Generations", &ga_cfg.num_generations)) {
                    ga_cfg.num_generations = std::max(1, ga_cfg.num_generations);
                }
                ImGui::PopItemWidth();

                // Row 2: Mutation Rate | Crossover Rate
                ImGui::PushItemWidth(100);
                if (ImGui::InputFloat("Mutation Rate", &ga_cfg.mutation_rate, 0.01f, 0.1f, "%.2f")) {
                    ga_cfg.mutation_rate = std::clamp(ga_cfg.mutation_rate, 0.0f, 1.0f);
                }
                ImGui::SameLine();
                if (ImGui::InputFloat("Crossover Rate", &ga_cfg.crossover_rate, 0.01f, 0.1f, "%.2f")) {
                    ga_cfg.crossover_rate = std::clamp(ga_cfg.crossover_rate, 0.0f, 1.0f);
                }
                ImGui::PopItemWidth();

                // Row 3: Frames per Evaluation
                ImGui::PushItemWidth(100);
                if (ImGui::InputInt("Frames/Eval", &ga_cfg.frames_per_eval)) {
                    ga_cfg.frames_per_eval = std::max(1, ga_cfg.frames_per_eval);
                }
                ImGui::PopItemWidth();

                ImGui::Separator();
                ImGui::Text("Parameters to Optimize:");

                ImGui::Checkbox("bias_diff##ga_opt", &ga_cfg.optimize_bias_diff); ImGui::SameLine();
                ImGui::Checkbox("bias_diff_on##ga_opt", &ga_cfg.optimize_bias_diff_on);
                ImGui::Checkbox("bias_diff_off##ga_opt", &ga_cfg.optimize_bias_diff_off); ImGui::SameLine();
                ImGui::Checkbox("bias_refr##ga_opt", &ga_cfg.optimize_bias_refr);
                ImGui::Checkbox("bias_fo##ga_opt", &ga_cfg.optimize_bias_fo); ImGui::SameLine();
                ImGui::Checkbox("bias_hpf##ga_opt", &ga_cfg.optimize_bias_hpf);
                ImGui::Checkbox("accumulation##ga_opt", &ga_cfg.optimize_accumulation);
                ImGui::Checkbox("trail_filter##ga_opt", &ga_cfg.optimize_trail_filter); ImGui::SameLine();
                ImGui::Checkbox("antiflicker##ga_opt", &ga_cfg.optimize_antiflicker);
                ImGui::Checkbox("erc##ga_opt", &ga_cfg.optimize_erc);

                ImGui::Separator();

                // Cluster filter settings
                settings_panel.render_genetic_algorithm();

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
                        params.alpha = 2.0f;
                        params.beta = 0.5f;
                        params.gamma = 2.0f;
                        params.minimum_event_pixels = 500;
                        params.delta = 5.0f;

                        // Set optimization mask from config checkboxes
                        params.opt_mask.bias_diff = ga_cfg.optimize_bias_diff;
                        params.opt_mask.bias_diff_on = ga_cfg.optimize_bias_diff_on;
                        params.opt_mask.bias_diff_off = ga_cfg.optimize_bias_diff_off;
                        params.opt_mask.bias_refr = ga_cfg.optimize_bias_refr;
                        params.opt_mask.bias_fo = ga_cfg.optimize_bias_fo;
                        params.opt_mask.bias_hpf = ga_cfg.optimize_bias_hpf;
                        params.opt_mask.accumulation = ga_cfg.optimize_accumulation;
                        params.opt_mask.trail_filter = ga_cfg.optimize_trail_filter;
                        params.opt_mask.antiflicker = ga_cfg.optimize_antiflicker;
                        params.opt_mask.erc = ga_cfg.optimize_erc;

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

                        auto fitness_callback = [&config, &ga_cfg, hw_ranges](const EventCameraGeneticOptimizer::Genome& genome) {
                            EventCameraGeneticOptimizer::Genome genome_copy = genome;
                            genome_copy.set_ranges(hw_ranges);
                            genome_copy.clamp();
                            return evaluate_genome_fitness(genome_copy, config, ga_cfg.frames_per_eval);
                        };

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

                        ga_state.optimizer = std::make_unique<EventCameraGeneticOptimizer>(
                            params, fitness_callback, progress_callback);

                        ga_state.optimizer_thread = std::make_unique<std::thread>([&]() {
                            std::cout << "GA optimization thread started" << std::endl;
                            EventCameraGeneticOptimizer::Genome best = ga_state.optimizer->optimize();
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
                    // Wait for thread to finish before allowing restart
                    if (ga_state.optimizer_thread && ga_state.optimizer_thread->joinable()) {
                        ga_state.optimizer_thread->join();
                    }
                    ga_state.running = false;
                    std::cout << "GA optimization stopped successfully" << std::endl;
                }
            }

            ImGui::Separator();

            // Display best results
            if (ga_state.best_fitness < 1e9f) {
                ImGui::Text("Best Results");
                ImGui::Text("Fitness: %.4f", ga_state.best_fitness.load());
                ImGui::Text("Contrast: %.2f", ga_state.best_result.contrast_score);
                ImGui::Text("Noise: %.4f", ga_state.best_result.noise_metric);

                if (!ga_state.running && app_state->camera_state().is_connected()) {
                    if (ImGui::Button("Apply Best Parameters", ImVec2(-1, 0))) {
                        EventCameraGeneticOptimizer::Genome clamped_genome = ga_state.best_genome;

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

                        config.camera_settings().bias_diff = clamped_genome.bias_diff;
                        config.camera_settings().bias_diff_on = clamped_genome.bias_diff_on;
                        config.camera_settings().bias_diff_off = clamped_genome.bias_diff_off;
                        config.camera_settings().bias_refr = clamped_genome.bias_refr;
                        config.camera_settings().bias_fo = clamped_genome.bias_fo;
                        config.camera_settings().bias_hpf = clamped_genome.bias_hpf;
                        config.camera_settings().accumulation_time_s = clamped_genome.accumulation_time_s;

                        int num_cameras = app_state->camera_state().num_cameras();
                        for (int i = 0; i < num_cameras; ++i) {
                            auto& cam_info = app_state->camera_state().camera_manager()->get_camera(i);
                            apply_bias_settings(*cam_info.camera, config.camera_settings());
                        }

                        std::cout << "Applied best GA parameters to all cameras (clamped to hardware limits)" << std::endl;
                    }
                }
            } else {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No optimization results yet");
            }
        }
        ImGui::End();

        // Strip 4: Camera Status
        ImGui::SetNextWindowPos(ImVec2(3 * strip_width + 4 * cam_spacing, settings_top), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(strip_width, settings_height), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Camera Status")) {
            // Display status for both cameras (left on top, right on bottom)
            for (int cam_idx = 0; cam_idx < 2; ++cam_idx) {
                if (cam_idx == 1) {
                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();
                }

                const char* cam_label = (cam_idx == 0) ? "Left Camera" : "Right Camera";
                ImGui::Text("%s", cam_label);
                ImGui::Separator();

                // Camera info
                if (app_state->camera_state().is_connected() && app_state->camera_state().camera_manager()) {
                    if (cam_idx < app_state->camera_state().camera_manager()->num_cameras()) {
                        auto& cam_info = app_state->camera_state().camera_manager()->get_camera(cam_idx);
                        ImGui::Text("Serial: %s", cam_info.serial.c_str());
                    } else {
                        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Not connected");
                    }
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Mode: SIMULATION");
                }

                // Resolution
                int width = app_state->display_settings().get_image_width();
                int height = app_state->display_settings().get_image_height();
                ImGui::Text("Resolution: %dx%d", width, height);

                // FPS display
                ImGuiIO& io = ImGui::GetIO();
                ImGui::Text("Display FPS: %.1f", io.Framerate);

                // Event rate display
                int64_t event_rate = app_state->event_metrics().get_events_per_second();
                if (event_rate > 1000000) {
                    ImGui::Text("Event rate: %.2f M events/sec", event_rate / 1000000.0f);
                } else if (event_rate > 1000) {
                    ImGui::Text("Event rate: %.1f K events/sec", event_rate / 1000.0f);
                } else {
                    ImGui::Text("Event rate: %lld events/sec", event_rate);
                }

                // Frame generation diagnostics
                int64_t gen = app_state->frame_buffer(cam_idx).get_frames_generated();
                int64_t drop = app_state->frame_buffer(cam_idx).get_frames_dropped();
                if (gen > 0) {
                    float drop_rate = (drop * 100.0f) / gen;
                    ImGui::Text("Frames: %lld gen, %lld drop (%.1f%%)", gen, drop, drop_rate);
                }

                // Event latency
                int64_t event_ts = app_state->event_metrics().get_last_event_timestamp();
                int64_t cam_start = app_state->camera_state().get_camera_start_time_us();
                if (event_ts > 0 && cam_start > 0) {
                    auto now = std::chrono::steady_clock::now();
                    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
                    int64_t event_system_ts = cam_start + event_ts;
                    float event_latency_ms = (now_us - event_system_ts) / 1000.0f;
                    ImGui::Text("Event latency: %.1f ms", event_latency_ms);
                }

                // Frame display latency
                auto now = std::chrono::steady_clock::now();
                auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
                int64_t frame_sys_ts = app_state->frame_sync(cam_idx).get_last_frame_system_ts();
                if (frame_sys_ts > 0) {
                    float frame_latency_ms = (now_us - frame_sys_ts) / 1000.0f;
                    ImGui::Text("Frame latency: %.1f ms", frame_latency_ms);
                }
            }
        }
        ImGui::End();

        // Strip 5: Controls
        ImGui::SetNextWindowPos(ImVec2(4 * strip_width + 5 * cam_spacing, settings_top), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(strip_width, settings_height), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Controls")) {
            // Connection buttons
            if (app_state->camera_state().is_connected()) {
                if (ImGui::Button("Disconnect & Reconnect", ImVec2(-1, 0))) {
                    settings_panel.set_camera_reconnect_request();
                }
            } else {
                if (ImGui::Button("Connect Camera", ImVec2(-1, 0))) {
                    settings_panel.set_camera_connect_request();
                }
            }

            if (ImGui::Button("Capture Frame", ImVec2(-1, 0))) {
                settings_panel.capture_frame();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            settings_panel.render_display_settings();

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            settings_panel.render_apply_button();
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

    // Wait for event thread (simulation mode only)
    if (app_state->camera_state().is_simulation_mode()) {
        if (app_state->camera_state().event_thread(0) && app_state->camera_state().event_thread(0)->joinable()) {
            app_state->camera_state().event_thread(0)->join();
        }
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
