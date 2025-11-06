#include "camera/features/trail_filter_feature.h"
#include <imgui.h>
#include <iostream>

namespace EventCamera {

bool TrailFilterFeature::initialize(Metavision::Camera& camera) {
    trail_filter_ = camera.get_device().get_facility<Metavision::I_EventTrailFilterModule>();

    if (!trail_filter_) {
        return false;
    }

    try {
        // Get reasonable defaults from hardware
        uint32_t min_thresh = trail_filter_->get_min_supported_threshold();
        uint32_t max_thresh = trail_filter_->get_max_supported_threshold();

        // Set default threshold
        threshold_us_ = std::min(10000, (int)max_thresh);

        std::cout << "TrailFilterFeature: Initialized (threshold range: " << min_thresh
                 << " - " << max_thresh << " μs)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "TrailFilterFeature: Warning during initialization: " << e.what() << std::endl;
    }

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
    if (!trail_filter_) return;

    enabled_ = enabled;

    try {
        trail_filter_->enable(enabled_);
        std::cout << "Event Trail Filter " << (enabled_ ? "enabled" : "disabled") << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "TrailFilterFeature: Failed to " << (enabled ? "enable" : "disable")
                 << ": " << e.what() << std::endl;
    }
}

void TrailFilterFeature::apply_settings() {
    if (!trail_filter_ || !enabled_) return;

    try {
        // Apply filter type
        Metavision::I_EventTrailFilterModule::Type type;
        switch (filter_type_) {
            case 0: type = Metavision::I_EventTrailFilterModule::Type::TRAIL; break;
            case 1: type = Metavision::I_EventTrailFilterModule::Type::STC_CUT_TRAIL; break;
            case 2: type = Metavision::I_EventTrailFilterModule::Type::STC_KEEP_TRAIL; break;
            default: type = Metavision::I_EventTrailFilterModule::Type::TRAIL; break;
        }
        trail_filter_->set_type(type);

        // Apply threshold
        trail_filter_->set_threshold(threshold_us_);

        std::cout << "Trail Filter settings applied: type=" << filter_type_
                 << " threshold=" << threshold_us_ << "μs" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "TrailFilterFeature: Failed to apply settings: " << e.what() << std::endl;
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

bool TrailFilterFeature::render_ui() {
    if (!is_available()) {
        return false;
    }

    bool changed = false;

    if (ImGui::CollapsingHeader("Event Trail Filter")) {
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

        // Threshold Delay
        ImGui::Text("Threshold Delay:");
        try {
            uint32_t min_thresh = trail_filter_->get_min_supported_threshold();
            uint32_t max_thresh = trail_filter_->get_max_supported_threshold();

            if (ImGui::SliderInt("Threshold (μs)", &threshold_us_, min_thresh, max_thresh)) {
                apply_settings();
                changed = true;
            }

            ImGui::TextWrapped("Range: %d - %d μs", min_thresh, max_thresh);
            ImGui::Spacing();
            ImGui::TextWrapped("Lower threshold = more aggressive filtering");
        } catch (const std::exception& e) {
            ImGui::TextWrapped("Error: Could not get threshold range");
        }
    }

    return changed;
}

} // namespace EventCamera
