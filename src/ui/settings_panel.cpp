#include "ui/settings_panel.h"
#include "core/app_state.h"
#include "camera_manager.h"
#include <imgui.h>
#include <iostream>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <filesystem>

namespace ui {

SettingsPanel::SettingsPanel(core::AppState& state,
                             AppConfig& config,
                             EventCamera::BiasManager& bias_mgr)
    : state_(state)
    , config_(config)
    , bias_mgr_(bias_mgr) {
    previous_settings_ = config_.camera_settings();
    // Set panel position and size (left side of screen)
    set_position(ImVec2(10, 10));
    set_size(ImVec2(400, 600));
}

void SettingsPanel::render() {
    if (!visible_) return;

    ImGui::SetNextWindowPos(position_, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(size_, ImGuiCond_FirstUseEver);

    if (ImGui::Begin(title().c_str(), &visible_)) {
        // Capture Frame button at the top
        if (ImGui::Button("Capture Frame", ImVec2(-1, 0))) {
            capture_frame();
        }
        ImGui::Separator();

        // ImageJ streaming controls
        ImGui::Text("ImageJ Streaming");
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
        ImGui::Separator();

        render_connection_controls();

        ImGui::Separator();
        render_bias_controls();

        ImGui::Separator();
        render_display_settings();

        ImGui::Separator();
        render_frame_generation();

        render_apply_button();
    }
    ImGui::End();
}

void SettingsPanel::render_connection_controls() {
    if (state_.camera_state().is_connected() && state_.camera_state().camera_manager()) {
        auto& cam_info = state_.camera_state().camera_manager()->get_camera(0);
        ImGui::Text("Camera: %s", cam_info.serial.c_str());

        if (ImGui::Button("Disconnect & Reconnect Camera", ImVec2(-1, 0))) {
            camera_reconnect_requested_ = true;
        }
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Mode: SIMULATION");
        if (ImGui::Button("Connect Camera", ImVec2(-1, 0))) {
            camera_connect_requested_ = true;
        }
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
            // Apply bias settings
            if (state_.camera_state().is_connected() && bias_mgr_.is_initialized()) {
                bias_mgr_.apply_to_camera();
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
        bias_mgr_.reset_to_defaults();
        config_.camera_settings().accumulation_time_s = 0.01f;
        settings_changed_ = true;
    }
}

void SettingsPanel::capture_frame() {
    std::cout << "Capture Frame button clicked!" << std::endl;

    // Get currently displayed frame from texture manager
    cv::Mat frame = state_.texture_manager().get_last_frame();
    std::cout << "Frame size: " << frame.cols << "x" << frame.rows
              << ", channels: " << frame.channels()
              << ", empty: " << (frame.empty() ? "YES" : "NO") << std::endl;

    if (!frame.empty()) {
        // Generate timestamped filename
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << "capture_"
           << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
           << "_" << std::setfill('0') << std::setw(3) << ms.count()
           << ".png";

        std::string filename = ss.str();

        // Construct full path with configured directory
        std::string capture_dir = config_.camera_settings().capture_directory;
        std::string full_path = capture_dir;
        if (!capture_dir.empty() && capture_dir.back() != '\\' && capture_dir.back() != '/') {
            full_path += "\\";
        }
        full_path += filename;

        std::cout << "Attempting to save to: " << full_path << std::endl;

        try {
            bool success = cv::imwrite(full_path, frame);
            if (success) {
                std::cout << "Frame captured successfully: " << full_path << std::endl;
            } else {
                std::cerr << "cv::imwrite returned false - failed to save: " << full_path << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception while capturing frame: " << e.what() << std::endl;
        }
    } else {
        std::cout << "No frame available to capture (frame is empty)" << std::endl;
    }
}

} // namespace ui
