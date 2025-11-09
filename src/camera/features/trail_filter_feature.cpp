#include "camera/features/trail_filter_feature.h"
#include <imgui.h>
#include <iostream>

namespace EventCamera {

bool TrailFilterFeature::initialize(Metavision::Camera& camera) {
    trail_filter_ = camera.get_device().get_facility<Metavision::I_EventTrailFilterModule>();

    if (!trail_filter_) {
        return false;
    }

    // Add to list of all cameras
    all_trail_filters_.clear();
    all_trail_filters_.push_back(trail_filter_);

    try {
        // Get reasonable defaults from hardware
        uint32_t min_thresh = trail_filter_->get_min_supported_threshold();
        uint32_t max_thresh = trail_filter_->get_max_supported_threshold();

        // Read current camera state (which was set from config in main.cpp)
        enabled_ = trail_filter_->is_enabled();
        threshold_us_ = trail_filter_->get_threshold();
        auto camera_type = trail_filter_->get_type();
        filter_type_ = (camera_type == Metavision::I_EventTrailFilterModule::Type::TRAIL) ? 0 :
                       (camera_type == Metavision::I_EventTrailFilterModule::Type::STC_CUT_TRAIL) ? 1 : 2;

        std::cout << "TrailFilterFeature: Initialized (threshold range: " << min_thresh
                 << " - " << max_thresh << " μs)" << std::endl;
        std::cout << "  Camera state: " << (enabled_ ? "enabled" : "disabled")
                 << ", type=" << filter_type_
                 << ", threshold=" << threshold_us_ << "μs" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "TrailFilterFeature: Warning during initialization: " << e.what() << std::endl;
    }

    return true;
}

bool TrailFilterFeature::add_camera(Metavision::Camera& camera) {
    auto* camera_trail_filter = camera.get_device().get_facility<Metavision::I_EventTrailFilterModule>();

    if (!camera_trail_filter) {
        std::cerr << "TrailFilterFeature: Additional camera does not support Trail Filter" << std::endl;
        return false;
    }

    all_trail_filters_.push_back(camera_trail_filter);
    std::cout << "TrailFilterFeature: Added camera (now controlling " << all_trail_filters_.size() << " cameras)" << std::endl;
    return true;
}

void TrailFilterFeature::shutdown() {
    if (trail_filter_ && enabled_) {
        try {
            trail_filter_->enable(false);
        } catch (...) {}
    }
    trail_filter_ = nullptr;
    enabled_ = false;
}

void TrailFilterFeature::enable(bool enabled) {
    if (all_trail_filters_.empty()) return;

    std::cout << "\n=== Trail Filter Toggle ===" << std::endl;
    std::cout << "Changing enabled state from " << (enabled_ ? "ON" : "OFF")
              << " to " << (enabled ? "ON" : "OFF") << std::endl;
    std::cout << "Current settings: type=" << filter_type_
              << ", threshold=" << threshold_us_ << "μs" << std::endl;

    // Query camera state BEFORE change
    for (size_t i = 0; i < all_trail_filters_.size(); ++i) {
        try {
            bool camera_enabled = all_trail_filters_[i]->is_enabled();
            uint32_t camera_threshold = all_trail_filters_[i]->get_threshold();
            auto camera_type = all_trail_filters_[i]->get_type();
            int camera_type_idx = (camera_type == Metavision::I_EventTrailFilterModule::Type::TRAIL) ? 0 :
                                  (camera_type == Metavision::I_EventTrailFilterModule::Type::STC_CUT_TRAIL) ? 1 : 2;
            std::cout << "  Camera " << i << " BEFORE: "
                     << (camera_enabled ? "enabled" : "disabled")
                     << ", type=" << camera_type_idx
                     << ", threshold=" << camera_threshold << "μs" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  Camera " << i << " BEFORE: Failed to query state: " << e.what() << std::endl;
        }
    }

    enabled_ = enabled;

    std::cout << "Event Trail Filter " << (enabled_ ? "enabling" : "disabling")
              << " on " << all_trail_filters_.size() << " camera(s)..." << std::endl;

    for (size_t i = 0; i < all_trail_filters_.size(); ++i) {
        try {
            all_trail_filters_[i]->enable(enabled_);
            std::cout << "  Camera " << i << ": " << (enabled_ ? "enabled" : "disabled") << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  Camera " << i << ": Failed to " << (enabled ? "enable" : "disable")
                     << ": " << e.what() << std::endl;
        }
    }

    // Query camera state AFTER change
    for (size_t i = 0; i < all_trail_filters_.size(); ++i) {
        try {
            bool camera_enabled = all_trail_filters_[i]->is_enabled();
            uint32_t camera_threshold = all_trail_filters_[i]->get_threshold();
            auto camera_type = all_trail_filters_[i]->get_type();
            int camera_type_idx = (camera_type == Metavision::I_EventTrailFilterModule::Type::TRAIL) ? 0 :
                                  (camera_type == Metavision::I_EventTrailFilterModule::Type::STC_CUT_TRAIL) ? 1 : 2;
            std::cout << "  Camera " << i << " AFTER: "
                     << (camera_enabled ? "enabled" : "disabled")
                     << ", type=" << camera_type_idx
                     << ", threshold=" << camera_threshold << "μs" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  Camera " << i << " AFTER: Failed to query state: " << e.what() << std::endl;
        }
    }
    std::cout << "==========================\n" << std::endl;
}

void TrailFilterFeature::apply_settings() {
    if (all_trail_filters_.empty() || !enabled_) return;

    // Determine filter type
    Metavision::I_EventTrailFilterModule::Type type;
    switch (filter_type_) {
        case 0: type = Metavision::I_EventTrailFilterModule::Type::TRAIL; break;
        case 1: type = Metavision::I_EventTrailFilterModule::Type::STC_CUT_TRAIL; break;
        case 2: type = Metavision::I_EventTrailFilterModule::Type::STC_KEEP_TRAIL; break;
        default: type = Metavision::I_EventTrailFilterModule::Type::TRAIL; break;
    }

    std::cout << "Trail Filter applying settings to " << all_trail_filters_.size() << " camera(s)..." << std::endl;

    for (size_t i = 0; i < all_trail_filters_.size(); ++i) {
        try {
            all_trail_filters_[i]->set_type(type);
            all_trail_filters_[i]->set_threshold(threshold_us_);
            std::cout << "  Camera " << i << ": type=" << filter_type_
                     << " threshold=" << threshold_us_ << "μs" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  Camera " << i << ": Failed to apply settings: " << e.what() << std::endl;
        }
    }
}

void TrailFilterFeature::set_type(Metavision::I_EventTrailFilterModule::Type type) {
    switch (type) {
        case Metavision::I_EventTrailFilterModule::Type::TRAIL:
            filter_type_ = 0; break;
        case Metavision::I_EventTrailFilterModule::Type::STC_CUT_TRAIL:
            filter_type_ = 1; break;
        case Metavision::I_EventTrailFilterModule::Type::STC_KEEP_TRAIL:
            filter_type_ = 2; break;
        default:
            filter_type_ = 0; break;
    }
    apply_settings();
}

void TrailFilterFeature::set_threshold(uint32_t threshold_us) {
    threshold_us_ = threshold_us;
    apply_settings();
}

void TrailFilterFeature::sync_from_camera() {
    if (!trail_filter_) return;

    try {
        // Read current camera state
        enabled_ = trail_filter_->is_enabled();
        threshold_us_ = trail_filter_->get_threshold();
        auto camera_type = trail_filter_->get_type();
        filter_type_ = (camera_type == Metavision::I_EventTrailFilterModule::Type::TRAIL) ? 0 :
                       (camera_type == Metavision::I_EventTrailFilterModule::Type::STC_CUT_TRAIL) ? 1 : 2;

        std::cout << "TrailFilterFeature: Synced from camera - "
                 << (enabled_ ? "enabled" : "disabled")
                 << ", type=" << filter_type_
                 << ", threshold=" << threshold_us_ << "μs" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "TrailFilterFeature: Failed to sync from camera: " << e.what() << std::endl;
    }
}

bool TrailFilterFeature::render_ui() {
    if (!is_available()) {
        return false;
    }

    bool changed = false;

    if (ImGui::TreeNode("Event Trail Filter")) {
        ImGui::TextWrapped("%s", description().c_str());
        ImGui::Spacing();

        // Enable/disable checkbox
        bool enabled_ui = enabled_;
        if (ImGui::Checkbox("Enable Trail Filter", &enabled_ui)) {
            enable(enabled_ui);
            changed = true;
        }

        ImGui::Spacing();

        // Filter Type
        const char* types[] = {
            "TRAIL (Keep first event)",
            "STC_CUT_TRAIL (Keep second event)",
            "STC_KEEP_TRAIL (Keep trailing events)"
        };
        if (ImGui::Combo("Filter Type", &filter_type_, types, 3)) {
            apply_settings();
            changed = true;
        }

        ImGui::Spacing();

        // Threshold Delay (converted to ms for display)
        ImGui::Text("Threshold Delay:");
        try {
            uint32_t min_thresh_us = trail_filter_->get_min_supported_threshold();
            uint32_t max_thresh_us = trail_filter_->get_max_supported_threshold();

            // Convert to ms for display (constrain slider to 1-100ms range)
            int min_thresh_ms = std::max(1, (int)(min_thresh_us / 1000));
            int max_thresh_ms = std::min(100, (int)(max_thresh_us / 1000));
            int threshold_ms = threshold_us_ / 1000;

            // Slider
            if (ImGui::SliderInt("Threshold (ms)", &threshold_ms, min_thresh_ms, max_thresh_ms)) {
                threshold_us_ = threshold_ms * 1000;  // Convert back to μs
                apply_settings();
                changed = true;
            }

            // Input box next to slider
            ImGui::SameLine();
            ImGui::SetNextItemWidth(80);
            int temp_ms = threshold_ms;
            if (ImGui::InputInt("##threshold_input", &temp_ms)) {
                temp_ms = std::clamp(temp_ms, min_thresh_ms, max_thresh_ms);
                threshold_us_ = temp_ms * 1000;  // Convert back to μs
                apply_settings();
                changed = true;
            }

            ImGui::TextWrapped("Current: %d ms (%d μs) [Range: %d - %d ms]",
                              threshold_ms, threshold_us_, min_thresh_ms, max_thresh_ms);
            ImGui::Spacing();
            ImGui::TextWrapped("Lower threshold = more aggressive filtering");
        } catch (const std::exception& e) {
            ImGui::TextWrapped("Error: Could not get threshold range");
        }

        ImGui::TreePop();
    }

    return changed;
}

} // namespace EventCamera
