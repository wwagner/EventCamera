#include "camera/features/antiflicker_feature.h"
#include <imgui.h>
#include <iostream>
#include <algorithm>

namespace EventCamera {

bool AntiFlickerFeature::initialize(Metavision::Camera& camera) {
    antiflicker_ = camera.get_device().get_facility<Metavision::I_AntiFlickerModule>();

    if (!antiflicker_) {
        return false;
    }

    // Add to list of all cameras
    all_antiflicker_.clear();
    all_antiflicker_.push_back(antiflicker_);

    try {
        // Get reasonable defaults from hardware
        uint32_t min_freq = antiflicker_->get_min_supported_frequency();
        uint32_t max_freq = antiflicker_->get_max_supported_frequency();

        // Set default frequency range for 100-150Hz
        low_freq_ = std::max((int)min_freq, 100);
        high_freq_ = std::min((int)max_freq, 150);

        std::cout << "AntiFlickerFeature: Initialized (freq range: " << min_freq
                 << " - " << max_freq << " Hz)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "AntiFlickerFeature: Warning during initialization: " << e.what() << std::endl;
    }

    return true;
}

bool AntiFlickerFeature::add_camera(Metavision::Camera& camera) {
    auto* camera_antiflicker = camera.get_device().get_facility<Metavision::I_AntiFlickerModule>();

    if (!camera_antiflicker) {
        std::cerr << "AntiFlickerFeature: Additional camera does not support Anti-Flicker" << std::endl;
        return false;
    }

    all_antiflicker_.push_back(camera_antiflicker);
    std::cout << "AntiFlickerFeature: Added camera (now controlling " << all_antiflicker_.size() << " cameras)" << std::endl;
    return true;
}

void AntiFlickerFeature::shutdown() {
    if (antiflicker_ && enabled_) {
        try {
            antiflicker_->enable(false);
        } catch (...) {}
    }
    antiflicker_ = nullptr;
    all_antiflicker_.clear();
    enabled_ = false;
}

void AntiFlickerFeature::enable(bool enabled) {
    if (all_antiflicker_.empty()) return;

    enabled_ = enabled;

    std::cout << "Anti-Flicker " << (enabled_ ? "enabling" : "disabling")
              << " on " << all_antiflicker_.size() << " camera(s)..." << std::endl;

    for (size_t i = 0; i < all_antiflicker_.size(); ++i) {
        try {
            all_antiflicker_[i]->enable(enabled_);
            std::cout << "  Camera " << i << ": " << (enabled_ ? "enabled" : "disabled") << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  Camera " << i << ": Failed to " << (enabled ? "enable" : "disable")
                     << ": " << e.what() << std::endl;
        }
    }
}

void AntiFlickerFeature::apply_settings() {
    if (all_antiflicker_.empty() || !enabled_) return;

    // Determine filtering mode
    auto mode = (mode_ == 0) ?
        Metavision::I_AntiFlickerModule::BAND_STOP :
        Metavision::I_AntiFlickerModule::BAND_PASS;

    std::cout << "Anti-Flicker applying settings to " << all_antiflicker_.size() << " camera(s)..." << std::endl;

    for (size_t i = 0; i < all_antiflicker_.size(); ++i) {
        try {
            all_antiflicker_[i]->set_filtering_mode(mode);
            all_antiflicker_[i]->set_frequency_band(low_freq_, high_freq_);
            all_antiflicker_[i]->set_duty_cycle(duty_cycle_);
            std::cout << "  Camera " << i << ": mode=" << mode_
                     << " freq=[" << low_freq_ << "," << high_freq_ << "]Hz"
                     << " duty=" << duty_cycle_ << "%" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  Camera " << i << ": Failed to apply settings: " << e.what() << std::endl;
        }
    }
}

void AntiFlickerFeature::set_filtering_mode(int mode) {
    mode_ = mode;
    apply_settings();
}

void AntiFlickerFeature::set_frequency_band(uint32_t low_freq, uint32_t high_freq) {
    low_freq_ = low_freq;
    high_freq_ = high_freq;
    apply_settings();
}

void AntiFlickerFeature::set_duty_cycle(uint32_t duty_cycle) {
    duty_cycle_ = duty_cycle;
    apply_settings();
}

bool AntiFlickerFeature::render_ui() {
    if (!is_available()) {
        return false;
    }

    bool changed = false;

    if (ImGui::TreeNode("Anti-Flicker Filter")) {
        ImGui::TextWrapped("%s", description().c_str());
        ImGui::Spacing();

        // Enable/disable checkbox
        bool enabled_ui = enabled_;
        if (ImGui::Checkbox("Enable Anti-Flicker", &enabled_ui)) {
            enable(enabled_ui);
            changed = true;
        }

        ImGui::Spacing();

        // Filter Mode
        const char* modes[] = { "BAND_STOP (Remove frequencies)", "BAND_PASS (Keep frequencies)" };
        if (ImGui::Combo("Filter Mode", &mode_, modes, 2)) {
            apply_settings();
            changed = true;
        }

        ImGui::Spacing();
        ImGui::Text("Frequency Band:");

        // Get supported frequency range
        uint32_t min_freq = antiflicker_->get_min_supported_frequency();
        uint32_t max_freq = antiflicker_->get_max_supported_frequency();

        bool freq_changed = false;
        freq_changed |= ImGui::SliderInt("Low Frequency (Hz)", &low_freq_, min_freq, max_freq);
        freq_changed |= ImGui::SliderInt("High Frequency (Hz)", &high_freq_, min_freq, max_freq);

        if (freq_changed) {
            // Ensure low < high
            if (low_freq_ >= high_freq_) {
                high_freq_ = low_freq_ + 1;
            }
            apply_settings();
            changed = true;
        }

        ImGui::Spacing();

        // Duty Cycle
        if (ImGui::SliderInt("Duty Cycle (%)", &duty_cycle_, 0, 100)) {
            apply_settings();
            changed = true;
        }

        ImGui::Spacing();
        ImGui::TextWrapped("Common presets:");
        ImGui::SameLine();
        if (ImGui::SmallButton("50Hz")) {
            low_freq_ = std::max((int)min_freq, 45);
            high_freq_ = std::min((int)max_freq, 55);
            apply_settings();
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("60Hz")) {
            low_freq_ = std::max((int)min_freq, 55);
            high_freq_ = std::min((int)max_freq, 65);
            apply_settings();
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("100Hz")) {
            low_freq_ = std::max((int)min_freq, 95);
            high_freq_ = std::min((int)max_freq, 105);
            apply_settings();
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("120Hz")) {
            low_freq_ = std::max((int)min_freq, 115);
            high_freq_ = std::min((int)max_freq, 125);
            apply_settings();
            changed = true;
        }

        ImGui::TextWrapped("Range: %d - %d Hz", min_freq, max_freq);

        ImGui::TreePop();
    }

    return changed;
}

} // namespace EventCamera
