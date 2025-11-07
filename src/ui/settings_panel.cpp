#include "ui/settings_panel.h"
#include "core/app_state.h"
#include "camera_manager.h"
#include "camera/features/trail_filter_feature.h"
#include "camera/features/erc_feature.h"
#include "camera/features/antiflicker_feature.h"
#include <imgui.h>
#include <iostream>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <metavision/hal/facilities/i_erc_module.h>
#include <metavision/hal/facilities/i_antiflicker_module.h>
#include <metavision/hal/facilities/i_event_trail_filter_module.h>
#include <metavision/hal/facilities/i_roi.h>
#include <metavision/hal/facilities/i_digital_crop.h>
#include <metavision/hal/facilities/i_monitoring.h>

namespace ui {

SettingsPanel::SettingsPanel(core::AppState& state,
                             AppConfig& config,
                             EventCamera::BiasManager& bias_mgr)
    : state_(state)
    , config_(config)
    , bias_mgr_(bias_mgr) {
    previous_settings_ = config_.camera_settings();
    // Set panel position and size (left side of screen) - larger to fit all sections
    set_position(ImVec2(10, 10));
    set_size(ImVec2(450, 900));
}

void SettingsPanel::render() {
    if (!visible_) return;

    ImGui::SetNextWindowPos(position_, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(size_, ImGuiCond_FirstUseEver);

    if (ImGui::Begin(title().c_str(), &visible_)) {
        // Top buttons
        if (state_.camera_state().is_connected()) {
            if (ImGui::Button("Disconnect & Reconnect Camera", ImVec2(-1, 0))) {
                camera_reconnect_requested_ = true;
            }
        } else {
            if (ImGui::Button("Connect Camera", ImVec2(-1, 0))) {
                camera_connect_requested_ = true;
            }
        }

        if (ImGui::Button("Capture Frame", ImVec2(-1, 0))) {
            capture_frame();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Camera status info (always visible, not collapsible)
        render_connection_controls();

        ImGui::Spacing();
        ImGui::Separator();

        // All settings as collapsible sections
        if (ImGui::CollapsingHeader("Analog Biases", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_bias_controls();
        }

        if (ImGui::CollapsingHeader("Digital Features")) {
            render_digital_features();
        }

        if (ImGui::CollapsingHeader("Display Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_display_settings();
        }

        if (ImGui::CollapsingHeader("Frame Generation", ImGuiTreeNodeFlags_DefaultOpen)) {
            render_frame_generation();
        }

        if (ImGui::CollapsingHeader("ImageJ Streaming")) {
            bool streaming_enabled = config_.camera_settings().imagej_streaming_enabled;
            if (ImGui::Checkbox("Enable Streaming", &streaming_enabled)) {
                config_.camera_settings().imagej_streaming_enabled = streaming_enabled;
                if (streaming_enabled) {
                    std::cout << "ImageJ streaming enabled (" << config_.camera_settings().imagej_stream_fps << " FPS)" << std::endl;
                    std::cout << "Stream directory: " << config_.camera_settings().imagej_stream_directory << std::endl;
                } else {
                    std::cout << "ImageJ streaming disabled" << std::endl;
                }
            }
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "(%d FPS to %s)",
                config_.camera_settings().imagej_stream_fps,
                config_.camera_settings().imagej_stream_directory.c_str());
        }

        if (ImGui::CollapsingHeader("Genetic Algorithm Optimization")) {
            render_genetic_algorithm();
        }

        ImGui::Spacing();
        ImGui::Separator();
        render_apply_button();
    }
    ImGui::End();
}

void SettingsPanel::render_connection_controls() {
    // Camera status (no buttons - those are at the top)
    if (state_.camera_state().is_connected() && state_.camera_state().camera_manager()) {
        auto& cam_info = state_.camera_state().camera_manager()->get_camera(0);
        ImGui::Text("Camera: %s", cam_info.serial.c_str());
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Mode: SIMULATION");
    }

    int width = state_.display_settings().get_image_width();
    int height = state_.display_settings().get_image_height();
    ImGui::Text("Resolution: %dx%d", width, height);

    // FPS display
    ImGuiIO& io = ImGui::GetIO();
    ImGui::Text("Display FPS: %.1f", io.Framerate);

    // Event rate display
    int64_t event_rate = state_.event_metrics().get_events_per_second();
    if (event_rate > 1000000) {
        ImGui::Text("Event rate: %.2f M events/sec", event_rate / 1000000.0f);
    } else if (event_rate > 1000) {
        ImGui::Text("Event rate: %.1f K events/sec", event_rate / 1000.0f);
    } else {
        ImGui::Text("Event rate: %lld events/sec", event_rate);
    }

    // Frame generation diagnostics
    int64_t gen = state_.frame_buffer().get_frames_generated();
    int64_t drop = state_.frame_buffer().get_frames_dropped();
    if (gen > 0) {
        float drop_rate = (drop * 100.0f) / gen;
        ImGui::Text("Frames: %lld generated, %lld dropped (%.1f%%)", gen, drop, drop_rate);
    }

    // Event latency
    int64_t event_ts = state_.event_metrics().get_last_event_timestamp();
    int64_t cam_start = state_.camera_state().get_camera_start_time_us();
    if (event_ts > 0 && cam_start > 0) {
        auto now = std::chrono::steady_clock::now();
        auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
        int64_t event_system_ts = cam_start + event_ts;
        float event_latency_ms = (now_us - event_system_ts) / 1000.0f;
        ImGui::Text("Event latency: %.1f ms", event_latency_ms);
        if (event_latency_ms > 500.0f) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "WARNING: Events are %.1f sec old!", event_latency_ms/1000.0f);
        }
    }

    // Frame display latency
    auto now = std::chrono::steady_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    int64_t frame_sys_ts = state_.frame_sync().get_last_frame_system_ts();
    if (frame_sys_ts > 0) {
        float frame_latency_ms = (now_us - frame_sys_ts) / 1000.0f;
        ImGui::Text("Frame display latency: %.1f ms", frame_latency_ms);
    }
}

void SettingsPanel::render_bias_controls() {
    ImGui::Text("Camera Biases");
    ImGui::Text("Adjust these to tune event detection");
    ImGui::Spacing();

    auto& cam_settings = config_.camera_settings();
    const auto& bias_ranges = bias_mgr_.get_bias_ranges();

    // Helper lambda to map slider position (0-100) to bias value exponentially
    auto exp_to_bias = [](float slider_pos, int min_val, int max_val) -> int {
        float normalized = slider_pos / 100.0f;
        float exponential = std::pow(normalized, 2.5f);
        return static_cast<int>(min_val + exponential * (max_val - min_val));
    };

    auto bias_to_exp = [](int bias_val, int min_val, int max_val) -> float {
        float normalized = (bias_val - min_val) / static_cast<float>(max_val - min_val);
        return std::pow(normalized, 1.0f / 2.5f) * 100.0f;
    };

    // Static variables to track slider positions (initialized once)
    static float slider_diff = 50.0f;
    static float slider_diff_on = 50.0f;
    static float slider_diff_off = 50.0f;
    static float slider_refr = 50.0f;
    static float slider_fo = 50.0f;
    static float slider_hpf = 50.0f;
    static bool sliders_initialized = false;

    // Initialize sliders from current bias values (only once)
    if (!sliders_initialized && !bias_ranges.empty()) {
        if (bias_ranges.count("bias_diff"))
            slider_diff = bias_to_exp(cam_settings.bias_diff, bias_ranges.at("bias_diff").min, bias_ranges.at("bias_diff").max);
        if (bias_ranges.count("bias_diff_on"))
            slider_diff_on = bias_to_exp(cam_settings.bias_diff_on, bias_ranges.at("bias_diff_on").min, bias_ranges.at("bias_diff_on").max);
        if (bias_ranges.count("bias_diff_off"))
            slider_diff_off = bias_to_exp(cam_settings.bias_diff_off, bias_ranges.at("bias_diff_off").min, bias_ranges.at("bias_diff_off").max);
        if (bias_ranges.count("bias_refr"))
            slider_refr = bias_to_exp(cam_settings.bias_refr, bias_ranges.at("bias_refr").min, bias_ranges.at("bias_refr").max);
        if (bias_ranges.count("bias_fo"))
            slider_fo = bias_to_exp(cam_settings.bias_fo, bias_ranges.at("bias_fo").min, bias_ranges.at("bias_fo").max);
        if (bias_ranges.count("bias_hpf"))
            slider_hpf = bias_to_exp(cam_settings.bias_hpf, bias_ranges.at("bias_hpf").min, bias_ranges.at("bias_hpf").max);
        sliders_initialized = true;
    }

    // Exponentially-scaled bias sliders with input boxes
    if (bias_ranges.count("bias_diff")) {
        const auto& range = bias_ranges.at("bias_diff");
        if (ImGui::SliderFloat("bias_diff", &slider_diff, 0.0f, 100.0f, "%.0f%%")) {
            cam_settings.bias_diff = exp_to_bias(slider_diff, range.min, range.max);
            settings_changed_ = true;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        int temp_diff = cam_settings.bias_diff;
        if (ImGui::InputInt("##bias_diff_input", &temp_diff)) {
            temp_diff = std::clamp(temp_diff, range.min, range.max);
            cam_settings.bias_diff = temp_diff;
            slider_diff = bias_to_exp(temp_diff, range.min, range.max);
            settings_changed_ = true;
        }
        ImGui::TextWrapped("Event threshold: %d [%d to %d] (exponential scale)",
                           cam_settings.bias_diff, range.min, range.max);
        ImGui::Spacing();
    }

    if (bias_ranges.count("bias_diff_on")) {
        const auto& range = bias_ranges.at("bias_diff_on");
        if (ImGui::SliderFloat("bias_diff_on", &slider_diff_on, 0.0f, 100.0f, "%.0f%%")) {
            cam_settings.bias_diff_on = exp_to_bias(slider_diff_on, range.min, range.max);
            settings_changed_ = true;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        int temp_diff_on = cam_settings.bias_diff_on;
        if (ImGui::InputInt("##bias_diff_on_input", &temp_diff_on)) {
            temp_diff_on = std::clamp(temp_diff_on, range.min, range.max);
            cam_settings.bias_diff_on = temp_diff_on;
            slider_diff_on = bias_to_exp(temp_diff_on, range.min, range.max);
            settings_changed_ = true;
        }
        ImGui::TextWrapped("ON threshold: %d [%d to %d] (exponential scale)",
                           cam_settings.bias_diff_on, range.min, range.max);
        ImGui::Spacing();
    }

    if (bias_ranges.count("bias_diff_off")) {
        const auto& range = bias_ranges.at("bias_diff_off");
        if (ImGui::SliderFloat("bias_diff_off", &slider_diff_off, 0.0f, 100.0f, "%.0f%%")) {
            cam_settings.bias_diff_off = exp_to_bias(slider_diff_off, range.min, range.max);
            settings_changed_ = true;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        int temp_diff_off = cam_settings.bias_diff_off;
        if (ImGui::InputInt("##bias_diff_off_input", &temp_diff_off)) {
            temp_diff_off = std::clamp(temp_diff_off, range.min, range.max);
            cam_settings.bias_diff_off = temp_diff_off;
            slider_diff_off = bias_to_exp(temp_diff_off, range.min, range.max);
            settings_changed_ = true;
        }
        ImGui::TextWrapped("OFF threshold: %d [%d to %d] (exponential scale)",
                           cam_settings.bias_diff_off, range.min, range.max);
        ImGui::Spacing();
    }

    if (bias_ranges.count("bias_refr")) {
        const auto& range = bias_ranges.at("bias_refr");
        if (ImGui::SliderFloat("bias_refr", &slider_refr, 0.0f, 100.0f, "%.0f%%")) {
            cam_settings.bias_refr = exp_to_bias(slider_refr, range.min, range.max);
            settings_changed_ = true;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        int temp_refr = cam_settings.bias_refr;
        if (ImGui::InputInt("##bias_refr_input", &temp_refr)) {
            temp_refr = std::clamp(temp_refr, range.min, range.max);
            cam_settings.bias_refr = temp_refr;
            slider_refr = bias_to_exp(temp_refr, range.min, range.max);
            settings_changed_ = true;
        }
        ImGui::TextWrapped("Refractory: %d [%d to %d] (exponential scale)",
                           cam_settings.bias_refr, range.min, range.max);
        ImGui::Spacing();
    }

    if (bias_ranges.count("bias_fo")) {
        const auto& range = bias_ranges.at("bias_fo");
        if (ImGui::SliderFloat("bias_fo", &slider_fo, 0.0f, 100.0f, "%.0f%%")) {
            cam_settings.bias_fo = exp_to_bias(slider_fo, range.min, range.max);
            settings_changed_ = true;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        int temp_fo = cam_settings.bias_fo;
        if (ImGui::InputInt("##bias_fo_input", &temp_fo)) {
            temp_fo = std::clamp(temp_fo, range.min, range.max);
            cam_settings.bias_fo = temp_fo;
            slider_fo = bias_to_exp(temp_fo, range.min, range.max);
            settings_changed_ = true;
        }
        ImGui::TextWrapped("Follower: %d [%d to %d] (exponential scale)",
                           cam_settings.bias_fo, range.min, range.max);
        ImGui::Spacing();
    }

    if (bias_ranges.count("bias_hpf")) {
        const auto& range = bias_ranges.at("bias_hpf");
        if (ImGui::SliderFloat("bias_hpf", &slider_hpf, 0.0f, 100.0f, "%.0f%%")) {
            cam_settings.bias_hpf = exp_to_bias(slider_hpf, range.min, range.max);
            settings_changed_ = true;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        int temp_hpf = cam_settings.bias_hpf;
        if (ImGui::InputInt("##bias_hpf_input", &temp_hpf)) {
            temp_hpf = std::clamp(temp_hpf, range.min, range.max);
            cam_settings.bias_hpf = temp_hpf;
            slider_hpf = bias_to_exp(temp_hpf, range.min, range.max);
            settings_changed_ = true;
        }
        ImGui::TextWrapped("High-pass: %d [%d to %d] (exponential scale)",
                           cam_settings.bias_hpf, range.min, range.max);
        ImGui::Spacing();
    }
}

void SettingsPanel::render_display_settings() {
    ImGui::Text("Display Settings");

    int display_fps = state_.display_settings().get_target_fps();
    if (ImGui::SliderInt("Target Display FPS", &display_fps, 1, 60)) {
        state_.display_settings().set_target_fps(display_fps);
    }
    ImGui::TextWrapped("Limit display updates (lower = less CPU)");

    ImGui::Spacing();

    // Frame subtraction toggle
    if (state_.subtraction_filter()) {
        bool diff_enabled = state_.subtraction_filter()->is_enabled();
        if (ImGui::Checkbox("Enable Frame Subtraction", &diff_enabled)) {
            state_.subtraction_filter()->set_enabled(diff_enabled);
            std::cout << "Frame subtraction " << (diff_enabled ? "enabled" : "disabled") << std::endl;
        }
    }
    ImGui::TextWrapped("Subtract successive frames to visualize motion/changes");
}

void SettingsPanel::render_frame_generation() {
    ImGui::Text("Frame Generation");

    auto& cam_settings = config_.camera_settings();

    if (ImGui::SliderFloat("Accumulation (s)", &cam_settings.accumulation_time_s,
                          0.001f, 0.1f, "%.3f")) {
        settings_changed_ = true;
    }
    ImGui::TextWrapped("Time to accumulate events (lower = more responsive)");
}

void SettingsPanel::render_apply_button() {
    ImGui::Spacing();
    ImGui::Separator();

    if (settings_changed_) {
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Settings changed!");

        if (ImGui::Button("Apply Settings", ImVec2(-1, 0))) {
            // Apply bias settings to ALL cameras
            if (state_.camera_state().is_connected() && state_.camera_state().camera_manager()) {
                int num_cameras = state_.camera_state().num_cameras();
                std::cout << "Applying settings to all " << num_cameras << " camera(s)..." << std::endl;

                for (int i = 0; i < num_cameras; ++i) {
                    auto& cam_info = state_.camera_state().camera_manager()->get_camera(i);
                    auto* ll_biases = cam_info.camera->get_device().get_facility<Metavision::I_LL_Biases>();

                    if (ll_biases) {
                        std::cout << "Applying biases to Camera " << i << "..." << std::endl;
                        auto& cam_settings = config_.camera_settings();

                        try { ll_biases->set("bias_diff", cam_settings.bias_diff); } catch (...) {}
                        try { ll_biases->set("bias_diff_on", cam_settings.bias_diff_on); } catch (...) {}
                        try { ll_biases->set("bias_diff_off", cam_settings.bias_diff_off); } catch (...) {}
                        try { ll_biases->set("bias_refr", cam_settings.bias_refr); } catch (...) {}
                        try { ll_biases->set("bias_fo", cam_settings.bias_fo); } catch (...) {}
                        try { ll_biases->set("bias_hpf", cam_settings.bias_hpf); } catch (...) {}

                        std::cout << "  Camera " << i << " biases applied" << std::endl;
                    }
                }

                // Apply digital features to all cameras directly
                std::cout << "Applying digital features to all cameras..." << std::endl;
                apply_digital_features_to_all_cameras();
            }

            // Update frame generation note
            auto& cam_settings = config_.camera_settings();
            if (cam_settings.accumulation_time_s != previous_settings_.accumulation_time_s) {
                std::cout << "Note: Frame accumulation time will take effect after reconnecting camera" << std::endl;
                std::cout << "Config updated to " << cam_settings.accumulation_time_s << "s" << std::endl;
            }

            previous_settings_ = cam_settings;
            settings_changed_ = false;
        }
    }

    ImGui::Spacing();

    if (ImGui::Button("Reset to Defaults", ImVec2(-1, 0))) {
        // Reset bias manager to get default values
        bias_mgr_.reset_to_defaults();

        // Apply default biases to ALL cameras
        if (state_.camera_state().is_connected() && state_.camera_state().camera_manager()) {
            int num_cameras = state_.camera_state().num_cameras();
            std::cout << "Resetting biases to defaults on all " << num_cameras << " camera(s)..." << std::endl;

            // Get default values from config after reset
            auto& cam_settings = config_.camera_settings();

            for (int i = 0; i < num_cameras; ++i) {
                auto& cam_info = state_.camera_state().camera_manager()->get_camera(i);
                auto* ll_biases = cam_info.camera->get_device().get_facility<Metavision::I_LL_Biases>();

                if (ll_biases) {
                    std::cout << "Resetting Camera " << i << " to defaults..." << std::endl;
                    try { ll_biases->set("bias_diff", cam_settings.bias_diff); } catch (...) {}
                    try { ll_biases->set("bias_diff_on", cam_settings.bias_diff_on); } catch (...) {}
                    try { ll_biases->set("bias_diff_off", cam_settings.bias_diff_off); } catch (...) {}
                    try { ll_biases->set("bias_refr", cam_settings.bias_refr); } catch (...) {}
                    try { ll_biases->set("bias_fo", cam_settings.bias_fo); } catch (...) {}
                    try { ll_biases->set("bias_hpf", cam_settings.bias_hpf); } catch (...) {}
                }
            }
        }

        config_.camera_settings().accumulation_time_s = 0.01f;
        settings_changed_ = true;
    }
}

void SettingsPanel::capture_frame() {
    std::cout << "Capture Frame button clicked!" << std::endl;

    // Generate timestamped filename base (shared by both cameras)
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << "capture_"
       << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
       << "_" << std::setfill('0') << std::setw(3) << ms.count();

    std::string filename_base = ss.str();

    // Construct base path with configured directory
    std::string capture_dir = config_.camera_settings().capture_directory;
    std::string base_path = capture_dir;
    if (!capture_dir.empty() && capture_dir.back() != '\\' && capture_dir.back() != '/') {
        base_path += "\\";
    }

    // Capture from all cameras
    int num_cameras = state_.camera_state().num_cameras();
    const char* suffixes[] = {"_left", "_right"};

    for (int i = 0; i < num_cameras && i < 2; ++i) {
        cv::Mat frame = state_.texture_manager(i).get_last_frame();

        if (!frame.empty()) {
            std::string full_path = base_path + filename_base + suffixes[i] + ".png";
            std::cout << "Attempting to save Camera " << i << " to: " << full_path << std::endl;

            try {
                bool success = cv::imwrite(full_path, frame);
                if (success) {
                    std::cout << "Camera " << i << " frame captured successfully: " << full_path << std::endl;
                } else {
                    std::cerr << "cv::imwrite returned false - failed to save Camera " << i << ": " << full_path << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception while capturing frame from Camera " << i << ": " << e.what() << std::endl;
            }
        } else {
            std::cout << "No frame available to capture from Camera " << i << " (frame is empty)" << std::endl;
        }
    }
}

void SettingsPanel::render_digital_features() {
    if (!state_.camera_state().is_connected() || !state_.camera_state().camera_manager()) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Connect camera to access digital features");
        return;
    }

    // Use FeatureManager to render all digital features
    state_.feature_manager().render_all_ui();

    // Debug: show if no features rendered
    // This will be visible if no features are available
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Note: Only available features shown above");

    /* OLD MANUAL CODE - REPLACED BY FEATUREMANAGER
    auto& cam_info = state_.camera_state().camera_manager()->get_camera(0);

    // ERC (Event Rate Controller)
    Metavision::I_ErcModule* erc = cam_info.camera->get_device().get_facility<Metavision::I_ErcModule>();
    if (erc) {
        if (ImGui::TreeNode("Event Rate Controller (ERC)")) {
            static bool erc_enabled = false;
            static int erc_rate_kev = 1000;
            static bool erc_initialized = false;

            if (!erc_initialized) {
                try {
                    uint32_t current_rate = erc->get_cd_event_rate();
                    erc_rate_kev = current_rate / 1000;
                    std::cout << "ERC: Synced with hardware - current rate: " << current_rate << " ev/s" << std::endl;
                    erc_initialized = true;
                } catch (const std::exception& e) {
                    std::cerr << "ERC: Could not read initial state: " << e.what() << std::endl;
                    erc_initialized = true;
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
                    erc_enabled = !erc_enabled;
                }
            }

            ImGui::Spacing();
            uint32_t min_rate = erc->get_min_supported_cd_event_rate() / 1000;
            uint32_t max_rate = erc->get_max_supported_cd_event_rate() / 1000;

            if (ImGui::SliderInt("Event Rate (kev/s)", &erc_rate_kev, min_rate, max_rate)) {
                try {
                    uint32_t rate_ev_s = erc_rate_kev * 1000;
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
            ImGui::TreePop();
        }
    }

    // Anti-Flicker Module
    Metavision::I_AntiFlickerModule* antiflicker = cam_info.camera->get_device().get_facility<Metavision::I_AntiFlickerModule>();
    if (antiflicker) {
        if (ImGui::TreeNode("Anti-Flicker Filter")) {
            static bool af_enabled = false;
            static int af_mode = 0;
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
                    af_enabled = !af_enabled;
                }
            }

            ImGui::Spacing();

            const char* af_modes[] = { "BAND_STOP (Remove frequencies)", "BAND_PASS (Keep frequencies)" };
            if (ImGui::Combo("Filter Mode", &af_mode, af_modes, 2)) {
                antiflicker->set_filtering_mode(af_mode == 0 ?
                    Metavision::I_AntiFlickerModule::BAND_STOP :
                    Metavision::I_AntiFlickerModule::BAND_PASS);
                std::cout << "Anti-Flicker mode set to " << af_modes[af_mode] << std::endl;
            }

            ImGui::Spacing();
            ImGui::Text("Frequency Band:");

            uint32_t min_freq = antiflicker->get_min_supported_frequency();
            uint32_t max_freq = antiflicker->get_max_supported_frequency();

            bool freq_changed = false;
            freq_changed |= ImGui::SliderInt("Low Frequency (Hz)", &af_low_freq, min_freq, max_freq);
            freq_changed |= ImGui::SliderInt("High Frequency (Hz)", &af_high_freq, min_freq, max_freq);

            if (freq_changed) {
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

            ImGui::TextWrapped("Range: %d - %d Hz", min_freq, max_freq);
            ImGui::TreePop();
        }
    }

    // Event Trail Filter Module
    Metavision::I_EventTrailFilterModule* trail_filter = cam_info.camera->get_device().get_facility<Metavision::I_EventTrailFilterModule>();
    if (trail_filter) {
        if (ImGui::TreeNode("Event Trail Filter")) {
            static bool etf_enabled = false;
            static int etf_type = 0;
            static int etf_threshold = 10000;

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
            ImGui::TreePop();
        }
    }

    // Digital Crop Module
    Metavision::I_DigitalCrop* digital_crop = cam_info.camera->get_device().get_facility<Metavision::I_DigitalCrop>();
    if (digital_crop) {
        if (ImGui::TreeNode("Digital Crop")) {
            static bool dc_enabled = false;
            static int dc_x = 0, dc_y = 0;
            static int dc_width = state_.display_settings().get_image_width();
            static int dc_height = state_.display_settings().get_image_height();

            ImGui::TextWrapped("Crop sensor output to reduce resolution and data volume");
            ImGui::Spacing();

            if (ImGui::Checkbox("Enable Digital Crop", &dc_enabled)) {
                try {
                    digital_crop->enable(dc_enabled);
                    std::cout << "Digital Crop " << (dc_enabled ? "enabled" : "disabled") << std::endl;

                    if (dc_enabled) {
                        int max_w = state_.display_settings().get_image_width() - dc_x;
                        int max_h = state_.display_settings().get_image_height() - dc_y;
                        dc_width = std::min(dc_width, max_w);
                        dc_height = std::min(dc_height, max_h);

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
                    dc_enabled = !dc_enabled;
                }
            }

            ImGui::Spacing();

            ImGui::Text("Crop Region:");
            bool dc_changed = false;
            dc_changed |= ImGui::SliderInt("X Position", &dc_x, 0, state_.display_settings().get_image_width() - 1);
            dc_changed |= ImGui::SliderInt("Y Position", &dc_y, 0, state_.display_settings().get_image_height() - 1);
            dc_changed |= ImGui::SliderInt("Width", &dc_width, 1, state_.display_settings().get_image_width());
            dc_changed |= ImGui::SliderInt("Height", &dc_height, 1, state_.display_settings().get_image_height());

            if (dc_changed && dc_enabled) {
                try {
                    int max_w = state_.display_settings().get_image_width() - dc_x;
                    int max_h = state_.display_settings().get_image_height() - dc_y;
                    dc_width = std::min(dc_width, max_w);
                    dc_height = std::min(dc_height, max_h);

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
            ImGui::TreePop();
        }
    }
    */
}

void SettingsPanel::render_genetic_algorithm() {
    auto& ga_cfg = config_.ga_settings();

    // Cluster-based fitness evaluation
    ImGui::Text("Cluster-Based Fitness Evaluation:");
    ImGui::Spacing();

    if (ImGui::Checkbox("Enable Cluster Filter", &ga_cfg.enable_cluster_filter)) {
        settings_changed_ = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Focus fitness evaluation on circular clusters, suppressing single-pixel noise elsewhere");
    }

    if (ga_cfg.enable_cluster_filter) {
        ImGui::Indent();

        // Cluster radius slider
        if (ImGui::SliderInt("Cluster Radius (pixels)", &ga_cfg.cluster_radius, 10, 100)) {
            settings_changed_ = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Radius of circular cluster regions in pixels");
        }

        ImGui::Spacing();
        ImGui::Text("Cluster Centers (x, y):");

        // Display existing clusters with remove buttons
        std::vector<int> to_remove;
        for (size_t i = 0; i < ga_cfg.cluster_centers.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));

            auto& center = ga_cfg.cluster_centers[i];
            ImGui::Text("  [%d] (%d, %d)", static_cast<int>(i), center.first, center.second);
            ImGui::SameLine();

            if (ImGui::SmallButton("Remove")) {
                to_remove.push_back(static_cast<int>(i));
                settings_changed_ = true;
            }

            ImGui::PopID();
        }

        // Remove marked clusters (in reverse order to maintain indices)
        for (auto it = to_remove.rbegin(); it != to_remove.rend(); ++it) {
            ga_cfg.cluster_centers.erase(ga_cfg.cluster_centers.begin() + *it);
        }

        // Add new cluster controls
        ImGui::Spacing();
        static int new_cluster_x = 320;  // Center of 640x480 image
        static int new_cluster_y = 240;

        ImGui::InputInt("New Cluster X", &new_cluster_x);
        ImGui::InputInt("New Cluster Y", &new_cluster_y);

        if (ImGui::Button("Add Cluster")) {
            ga_cfg.cluster_centers.emplace_back(new_cluster_x, new_cluster_y);
            settings_changed_ = true;
        }

        ImGui::SameLine();
        if (ImGui::Button("Clear All Clusters")) {
            ga_cfg.cluster_centers.clear();
            settings_changed_ = true;
        }

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.0f, 1.0f),
                          "Tip: Add clusters at locations where you expect");
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.0f, 1.0f),
                          "events (e.g., LEDs, moving objects)");

        ImGui::Unindent();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                      "Other GA settings can be configured in event_config.ini");
}

void SettingsPanel::apply_digital_features_to_all_cameras() {
    if (!state_.camera_state().is_connected() || !state_.camera_state().camera_manager()) {
        return;
    }

    // Simply iterate through all features and apply them to Camera 0 (which feature manager is connected to)
    // Then manually sync the settings to other cameras
    std::cout << "Applying digital features to all cameras..." << std::endl;
    state_.feature_manager().apply_all_settings();

    // Now we need to manually copy the same settings to other cameras
    // Get Camera 0's feature states
    auto& cam0 = state_.camera_state().camera_manager()->get_camera(0);
    int num_cameras = state_.camera_state().num_cameras();

    for (int i = 1; i < num_cameras; ++i) {
        auto& cam = state_.camera_state().camera_manager()->get_camera(i);
        std::cout << "Syncing features from Camera 0 to Camera " << i << "..." << std::endl;

        // Sync Trail Filter
        auto* trail_filter_src = cam0.camera->get_device().get_facility<Metavision::I_EventTrailFilterModule>();
        auto* trail_filter_dst = cam.camera->get_device().get_facility<Metavision::I_EventTrailFilterModule>();
        if (trail_filter_src && trail_filter_dst) {
            try {
                trail_filter_dst->set_type(trail_filter_src->get_type());
                trail_filter_dst->set_threshold(trail_filter_src->get_threshold());
                trail_filter_dst->enable(trail_filter_src->is_enabled());
                std::cout << "  Trail Filter synced to Camera " << i << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "  Failed to sync Trail Filter to Camera " << i << ": " << e.what() << std::endl;
            }
        }

        // Sync ERC
        auto* erc_src = cam0.camera->get_device().get_facility<Metavision::I_ErcModule>();
        auto* erc_dst = cam.camera->get_device().get_facility<Metavision::I_ErcModule>();
        if (erc_src && erc_dst) {
            try {
                erc_dst->enable(erc_src->is_enabled());
                std::cout << "  ERC synced to Camera " << i << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "  Failed to sync ERC to Camera " << i << ": " << e.what() << std::endl;
            }
        }

        // Sync Anti-Flicker
        auto* antiflicker_src = cam0.camera->get_device().get_facility<Metavision::I_AntiFlickerModule>();
        auto* antiflicker_dst = cam.camera->get_device().get_facility<Metavision::I_AntiFlickerModule>();
        if (antiflicker_src && antiflicker_dst) {
            try {
                antiflicker_dst->enable(antiflicker_src->is_enabled());
                std::cout << "  Anti-Flicker synced to Camera " << i << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "  Failed to sync Anti-Flicker to Camera " << i << ": " << e.what() << std::endl;
            }
        }
    }
}

} // namespace ui
