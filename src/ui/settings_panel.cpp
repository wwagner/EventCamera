#include "ui/settings_panel.h"
#include "core/app_state.h"
#include <imgui.h>
#include <iostream>

namespace ui {

SettingsPanel::SettingsPanel(core::AppState& state,
                             AppConfig& config,
                             EventCamera::BiasManager& bias_mgr)
    : state_(state), config_(config), bias_mgr_(bias_mgr) {

    previous_settings_ = config_.camera_settings();
}

void SettingsPanel::render() {
    if (!visible_) return;

    ImGui::SetNextWindowPos(position_, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(size_, ImGuiCond_FirstUseEver);

    if (ImGui::Begin(title().c_str(), &visible_)) {
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
    ImGui::Text("Camera Connection");
    ImGui::Spacing();

    if (state_.camera_state().is_connected()) {
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Status: CONNECTED");

        if (ImGui::Button("Reconnect Camera", ImVec2(-1, 0))) {
            std::cout << "\n=== Reconnecting camera ===" << std::endl;
            // Note: Actual reconnection logic would need to be handled in main
            // This is a placeholder for the UI
        }
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Mode: SIMULATION");

        if (ImGui::Button("Connect Camera", ImVec2(-1, 0))) {
            std::cout << "Connect camera requested" << std::endl;
            // Note: Actual connection logic would need to be handled in main
        }
    }
}

void SettingsPanel::render_bias_controls() {
    ImGui::Text("Camera Biases");
    ImGui::Text("Adjust these to tune event detection");
    ImGui::Spacing();

    // Use BiasManager to render bias controls
    if (bias_mgr_.render_bias_ui("bias_diff", "bias_diff", "Event threshold")) {
        settings_changed_ = true;
    }

    if (bias_mgr_.render_bias_ui("bias_refr", "bias_refr", "Refractory")) {
        settings_changed_ = true;
    }

    if (bias_mgr_.render_bias_ui("bias_fo", "bias_fo", "Follower")) {
        settings_changed_ = true;
    }

    if (bias_mgr_.render_bias_ui("bias_hpf", "bias_hpf", "High-pass")) {
        settings_changed_ = true;
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

} // namespace ui
