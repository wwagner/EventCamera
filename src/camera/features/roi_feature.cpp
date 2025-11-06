#include "camera/features/roi_feature.h"
#include "video/filters/roi_filter.h"
#include <imgui.h>
#include <iostream>
#include <algorithm>

namespace EventCamera {

ROIFeature::ROIFeature(std::shared_ptr<video::ROIFilter> roi_filter, core::DisplaySettings& display_settings)
    : roi_filter_(roi_filter), display_settings_(display_settings) {

    // Initialize with reasonable defaults
    width_ = display_settings_.get_image_width() / 2;
    height_ = display_settings_.get_image_height() / 2;
}

bool ROIFeature::initialize(Metavision::Camera& camera) {
    roi_ = camera.get_device().get_facility<Metavision::I_ROI>();

    if (!roi_) {
        return false;
    }

    std::cout << "ROIFeature: Initialized" << std::endl;
    return true;
}

void ROIFeature::shutdown() {
    if (roi_ && enabled_) {
        try {
            roi_->enable(false);
        } catch (...) {}
    }
    roi_ = nullptr;
    enabled_ = false;
}

void ROIFeature::enable(bool enabled) {
    if (!roi_) return;

    enabled_ = enabled;

    try {
        roi_->enable(enabled_);
        std::cout << "ROI " << (enabled_ ? "enabled" : "disabled") << std::endl;

        // If enabling, apply current window
        if (enabled_) {
            apply_settings();
        }
    } catch (const std::exception& e) {
        std::cerr << "ROIFeature: Failed to " << (enabled ? "enable" : "disable")
                 << ": " << e.what() << std::endl;
    }
}

void ROIFeature::apply_settings() {
    if (!roi_ || !enabled_) return;

    try {
        // Clamp values to image dimensions
        int max_width = display_settings_.get_image_width();
        int max_height = display_settings_.get_image_height();

        width_ = std::min(width_, max_width - x_);
        height_ = std::min(height_, max_height - y_);

        Metavision::I_ROI::Window window(x_, y_, width_, height_);
        roi_->set_window(window);

        std::cout << "[ROI] Window set to: x=" << x_ << " y=" << y_
                 << " w=" << width_ << " h=" << height_ << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ROIFeature: Failed to apply settings: " << e.what() << std::endl;
    }
}

void ROIFeature::set_mode(Metavision::I_ROI::Mode mode) {
    if (!roi_) return;

    mode_ = (mode == Metavision::I_ROI::Mode::ROI) ? 0 : 1;

    try {
        roi_->set_mode(mode);
        std::cout << "ROI mode set to " << (mode_ == 0 ? "ROI" : "RONI") << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ROIFeature: Failed to set mode: " << e.what() << std::endl;
    }
}

void ROIFeature::set_window(int x, int y, int width, int height) {
    x_ = x;
    y_ = y;
    width_ = width;
    height_ = height;
    apply_settings();
}

bool ROIFeature::render_ui() {
    if (!is_available()) {
        return false;
    }

    bool changed = false;

    if (ImGui::CollapsingHeader("Region of Interest (ROI)")) {
        ImGui::TextWrapped("%s", description().c_str());
        ImGui::Spacing();

        // Enable/disable checkbox
        bool enabled_ui = enabled_;
        if (ImGui::Checkbox("Enable ROI", &enabled_ui)) {
            enable(enabled_ui);
            changed = true;
        }

        ImGui::SameLine();
        if (ImGui::Checkbox("Crop View to ROI", &crop_view_)) {
            changed = true;
        }

        ImGui::Spacing();

        // Mode selection
        const char* modes[] = { "ROI (Keep Inside)", "RONI (Discard Inside)" };
        if (ImGui::Combo("Mode", &mode_, modes, 2)) {
            set_mode(mode_ == 0 ? Metavision::I_ROI::Mode::ROI : Metavision::I_ROI::Mode::RONI);
            changed = true;
        }

        ImGui::Spacing();
        ImGui::Text("Window Position & Size:");

        // Position and size controls
        int max_width = display_settings_.get_image_width();
        int max_height = display_settings_.get_image_height();

        bool window_changed = false;
        window_changed |= ImGui::SliderInt("X", &x_, 0, max_width - 1);
        window_changed |= ImGui::SliderInt("Y", &y_, 0, max_height - 1);
        window_changed |= ImGui::SliderInt("Width", &width_, 1, max_width);
        window_changed |= ImGui::SliderInt("Height", &height_, 1, max_height);

        // Update visualization in real-time (always, even if ROI not enabled)
        if (roi_filter_) {
            roi_filter_->set_enabled(enabled_);
            roi_filter_->set_crop_to_roi(crop_view_);
            roi_filter_->set_show_rectangle(enabled_);
            roi_filter_->set_roi(x_, y_, width_, height_);
        }

        // Auto-apply if ROI is enabled and window changed
        if (window_changed && enabled_) {
            apply_settings();
            changed = true;
        }

        ImGui::TextWrapped("Window: [%d, %d] %dx%d", x_, y_, width_, height_);
        if (enabled_) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "ACTIVE");
        }
    }

    return changed;
}

} // namespace EventCamera
