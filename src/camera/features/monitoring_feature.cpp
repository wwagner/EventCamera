#include "camera/features/monitoring_feature.h"
#include <imgui.h>
#include <iostream>

namespace EventCamera {

bool MonitoringFeature::initialize(Metavision::Camera& camera) {
    monitoring_ = camera.get_device().get_facility<Metavision::I_Monitoring>();

    if (!monitoring_) {
        return false;
    }

    // Add to list of all cameras
    all_monitoring_.clear();
    all_monitoring_.push_back(monitoring_);

    // Detect which capabilities are actually supported
    detect_capabilities();

    return has_temperature_ || has_illumination_ || has_dead_time_;
}

bool MonitoringFeature::add_camera(Metavision::Camera& camera) {
    auto* camera_monitoring = camera.get_device().get_facility<Metavision::I_Monitoring>();

    if (!camera_monitoring) {
        std::cerr << "MonitoringFeature: Additional camera does not support Monitoring" << std::endl;
        return false;
    }

    all_monitoring_.push_back(camera_monitoring);
    std::cout << "MonitoringFeature: Added camera (now monitoring " << all_monitoring_.size() << " cameras)" << std::endl;
    return true;
}

void MonitoringFeature::shutdown() {
    monitoring_ = nullptr;
    all_monitoring_.clear();
    has_temperature_ = false;
    has_illumination_ = false;
    has_dead_time_ = false;
}

bool MonitoringFeature::is_available() const {
    return monitoring_ != nullptr &&
           (has_temperature_ || has_illumination_ || has_dead_time_);
}

void MonitoringFeature::detect_capabilities() {
    if (!monitoring_) return;

    std::cout << "MonitoringFeature: Checking capabilities..." << std::endl;

    // Temperature
    try {
        int temp = monitoring_->get_temperature();
        // If we got here without exception and temp is reasonable, mark as supported
        if (temp >= -40 && temp <= 120) {  // Reasonable sensor temp range
            has_temperature_ = true;
            std::cout << "  Temperature: supported (current: " << temp << "°C)" << std::endl;
        }
    } catch (...) {
        std::cout << "  Temperature: not supported" << std::endl;
    }

    // Illumination - FORCE DISABLED due to error spam on some cameras
    // Even if it "works", it generates HAL errors, so disable it
    has_illumination_ = false;
    std::cout << "  Illumination: disabled (generates errors on some cameras)" << std::endl;

    // Pixel Dead Time
    try {
        int dt = monitoring_->get_pixel_dead_time();
        if (dt >= 0 && dt <= 100000) {  // Reasonable dead time range (0-100ms)
            has_dead_time_ = true;
            std::cout << "  Pixel Dead Time: supported (current: " << dt << " μs)" << std::endl;
        }
    } catch (...) {
        std::cout << "  Pixel Dead Time: not supported" << std::endl;
    }
}

bool MonitoringFeature::render_ui() {
    if (!is_available()) {
        return false;
    }

    if (ImGui::TreeNode("Hardware Monitoring")) {
        // Display information for each camera
        for (size_t cam_idx = 0; cam_idx < all_monitoring_.size(); ++cam_idx) {
            auto* mon = all_monitoring_[cam_idx];

            if (all_monitoring_.size() > 1) {
                ImGui::Text("Camera %zu:", cam_idx);
                ImGui::Indent();
            }

            if (has_temperature_) {
                try {
                    int temp = mon->get_temperature();
                    ImGui::Text("Temperature: %d°C", temp);
                    if (temp > 60) {
                        ImGui::SameLine();
                        ImGui::TextColored(ImVec4(1, 0, 0, 1), "⚠ HOT");
                    }
                } catch (...) {
                    ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Temperature: Error");
                }
            }

            if (has_illumination_) {
                try {
                    int illum = mon->get_illumination();
                    ImGui::Text("Illumination: %d lux", illum);
                } catch (...) {
                    ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Illumination: Error");
                }
            }

            if (has_dead_time_) {
                try {
                    int dead_time = mon->get_pixel_dead_time();
                    ImGui::Text("Pixel Dead Time: %d μs", dead_time);
                } catch (...) {
                    ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "Pixel Dead Time: Error");
                }
            }

            if (all_monitoring_.size() > 1) {
                ImGui::Unindent();
                if (cam_idx < all_monitoring_.size() - 1) {
                    ImGui::Spacing();
                }
            }
        }

        if (!has_temperature_ && !has_illumination_ && !has_dead_time_) {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1),
                             "No monitoring features available");
        }

        ImGui::TreePop();
    }

    return false;  // Read-only feature, never changes settings
}

} // namespace EventCamera
