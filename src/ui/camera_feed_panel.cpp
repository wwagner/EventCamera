#include "ui/camera_feed_panel.h"
#include "core/app_state.h"
#include <imgui.h>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace ui {

CameraFeedPanel::CameraFeedPanel(core::AppState& state)
    : state_(state) {

    // Set larger default size for feed panel
    size_ = ImVec2(800, 650);
    position_ = ImVec2(440, 10);
}

void CameraFeedPanel::render() {
    if (!visible_) return;

    // Calculate window size to maintain camera aspect ratio
    float camera_aspect = static_cast<float>(state_.display_settings().get_image_width()) /
                         state_.display_settings().get_image_height();
    float feed_window_width = 800;
    float feed_window_height = (feed_window_width / camera_aspect) + 50;  // +50 for window chrome

    ImGui::SetNextWindowPos(position_, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(feed_window_width, feed_window_height),
                            ImGuiCond_FirstUseEver);

    if (ImGui::Begin(title().c_str(), &visible_)) {
        render_statistics();
        ImGui::Separator();
        render_feed_texture();
    }
    ImGui::End();
}

void CameraFeedPanel::render_statistics() {
    auto& io = ImGui::GetIO();

    // Resolution and FPS
    int width = state_.display_settings().get_image_width();
    int height = state_.display_settings().get_image_height();
    ImGui::Text("Resolution: %dx%d", width, height);
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

    // Frame generation and latency diagnostics
    int64_t gen = state_.frame_buffer().get_frames_generated();
    int64_t drop = state_.frame_buffer().get_frames_dropped();
    if (gen > 0) {
        float drop_rate = (drop * 100.0f) / gen;
        ImGui::Text("Frames: %lld generated, %lld dropped (%.1f%%)", gen, drop, drop_rate);
    }

    // Calculate event latency (how old are events when we receive them)
    int64_t event_ts = state_.event_metrics().get_last_event_timestamp();
    int64_t cam_start = state_.camera_state().get_camera_start_time_us();
    if (event_ts > 0 && cam_start > 0) {
        auto now = std::chrono::steady_clock::now();
        auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
        // Event timestamp is microseconds since camera start
        // Convert to system time by adding to camera start time
        int64_t event_system_ts = cam_start + event_ts;
        float event_latency_ms = (now_us - event_system_ts) / 1000.0f;
        ImGui::Text("Event latency: %.1f ms", event_latency_ms);
        if (event_latency_ms > 500.0f) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f),
                             "WARNING: Events are %.1f sec old!", event_latency_ms/1000.0f);
        }
    }
}

void CameraFeedPanel::render_feed_texture() {
    if (state_.texture_manager().get_texture_id() != 0) {
        ImVec2 window_size = ImGui::GetContentRegionAvail();

        // Reserve space for button at bottom
        float button_height = 30.0f;
        window_size.y -= button_height;

        // Maintain aspect ratio
        float aspect = static_cast<float>(state_.texture_manager().get_width()) /
                      state_.texture_manager().get_height();
        float display_width = window_size.x;
        float display_height = display_width / aspect;

        if (display_height > window_size.y) {
            display_height = window_size.y;
            display_width = display_height * aspect;
        }

        ImGui::Image((void*)(intptr_t)state_.texture_manager().get_texture_id(),
                   ImVec2(display_width, display_height));

        // Capture button below the image
        ImGui::Spacing();
        if (ImGui::Button("Capture Frame", ImVec2(-1, 0))) {
            capture_frame();
        }
    } else {
        ImGui::Text("Waiting for camera frames...");
    }
}

void CameraFeedPanel::capture_frame() {
    // Get current frame from frame buffer
    auto frame_opt = state_.frame_buffer().consume_frame();

    if (!frame_opt.has_value()) {
        std::cout << "No frame available to capture" << std::endl;
        return;
    }

    cv::Mat frame = frame_opt.value();
    if (frame.empty()) {
        std::cout << "Frame is empty, cannot capture" << std::endl;
        return;
    }

    // Generate filename with timestamp
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

    // Save frame to file
    try {
        cv::imwrite(filename, frame);
        std::cout << "Frame captured: " << filename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to capture frame: " << e.what() << std::endl;
    }
}

} // namespace ui
