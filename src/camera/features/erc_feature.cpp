#include "camera/features/erc_feature.h"
#include <imgui.h>
#include <iostream>

namespace EventCamera {

bool ERCFeature::initialize(Metavision::Camera& camera) {
    erc_ = camera.get_device().get_facility<Metavision::I_ErcModule>();

    if (!erc_) {
        return false;
    }

    // Query current settings
    try {
        enabled_ = erc_->is_enabled();

        // Get reasonable defaults from hardware ranges
        uint32_t min_rate = erc_->get_min_supported_cd_event_rate();
        uint32_t max_rate = erc_->get_max_supported_cd_event_rate();

        // Set initial rate to middle of range if not already set
        target_rate_kevps_ = (min_rate / 1000 + max_rate / 1000) / 2;

        std::cout << "ERCFeature: Initialized (rate range: " << min_rate / 1000
                 << " - " << max_rate / 1000 << " kev/s)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERCFeature: Warning during initialization: " << e.what() << std::endl;
    }

    return true;
}

void ERCFeature::shutdown() {
    if (erc_ && enabled_) {
        try {
            erc_->enable(false);
        } catch (...) {}
    }
    erc_ = nullptr;
    enabled_ = false;
}

void ERCFeature::enable(bool enabled) {
    if (!erc_) return;

    enabled_ = enabled;

    try {
        erc_->enable(enabled_);
        std::cout << "ERC " << (enabled_ ? "enabled" : "disabled") << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERCFeature: Failed to " << (enabled ? "enable" : "disable")
                 << ": " << e.what() << std::endl;
    }
}

void ERCFeature::apply_settings() {
    if (!erc_ || !enabled_) return;

    try {
        uint32_t rate_ev_s = target_rate_kevps_ * 1000;
        erc_->set_cd_event_rate(rate_ev_s);
        std::cout << "ERC rate set to " << rate_ev_s << " ev/s ("
                 << target_rate_kevps_ << " kev/s)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERCFeature: Failed to set event rate: " << e.what() << std::endl;
    }
}

void ERCFeature::set_event_rate_kevps(int rate_kevps) {
    target_rate_kevps_ = rate_kevps;
    apply_settings();
}

bool ERCFeature::render_ui() {
    if (!is_available()) {
        return false;
    }

    bool changed = false;

    if (ImGui::CollapsingHeader("Event Rate Controller (ERC)")) {
        ImGui::TextWrapped("%s", description().c_str());
        ImGui::Spacing();

        // Enable/disable checkbox
        bool enabled_ui = enabled_;
        if (ImGui::Checkbox("Enable ERC", &enabled_ui)) {
            enable(enabled_ui);
            changed = true;
        }

        ImGui::Spacing();

        if (enabled_) {
            // Get hardware limits
            uint32_t min_rate = erc_->get_min_supported_cd_event_rate() / 1000;  // kev/s
            uint32_t max_rate = erc_->get_max_supported_cd_event_rate() / 1000;

            // Rate slider
            int rate = target_rate_kevps_;
            if (ImGui::SliderInt("Event Rate (kev/s)", &rate, min_rate, max_rate)) {
                target_rate_kevps_ = rate;
                apply_settings();
                changed = true;
            }

            ImGui::TextWrapped("Current: %d kev/s (%d Mev/s)",
                             target_rate_kevps_, target_rate_kevps_ / 1000);
            ImGui::TextWrapped("Range: %d - %d kev/s", min_rate, max_rate);

            // Show count period
            try {
                uint32_t period = erc_->get_count_period();
                ImGui::Text("Count Period: %d μs", period);
            } catch (...) {}
        }
    }

    return changed;
}

} // namespace EventCamera
