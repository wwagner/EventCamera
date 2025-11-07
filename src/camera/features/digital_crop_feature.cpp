#include "camera/features/digital_crop_feature.h"
#include <imgui.h>
#include <iostream>
#include <algorithm>

namespace EventCamera {

DigitalCropFeature::DigitalCropFeature(core::DisplaySettings& display_settings)
    : display_settings_(display_settings) {

    // Initialize with full sensor size
    width_ = display_settings_.get_image_width();
    height_ = display_settings_.get_image_height();
}

bool DigitalCropFeature::initialize(Metavision::Camera& camera) {
    digital_crop_ = camera.get_device().get_facility<Metavision::I_DigitalCrop>();

    if (!digital_crop_) {
        return false;
    }

    std::cout << "DigitalCropFeature: Initialized" << std::endl;
    return true;
}

void DigitalCropFeature::shutdown() {
    if (digital_crop_ && enabled_) {
        try {
            digital_crop_->enable(false);
        } catch (...) {}
    }
    digital_crop_ = nullptr;
    enabled_ = false;
}

void DigitalCropFeature::enable(bool enabled) {
    if (!digital_crop_) return;

    enabled_ = enabled;

    try {
        digital_crop_->enable(enabled_);
        std::cout << "Digital Crop " << (enabled_ ? "enabled" : "disabled") << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "DigitalCropFeature: Failed to " << (enabled ? "enable" : "disable")
                 << ": " << e.what() << std::endl;
    }
}

void DigitalCropFeature::apply_settings() {
    if (!digital_crop_ || !enabled_) return;

    try {
        // Clamp values to image dimensions
        int max_width = display_settings_.get_image_width();
        int max_height = display_settings_.get_image_height();

        width_ = std::min(width_, max_width - x_);
        height_ = std::min(height_, max_height - y_);

        // Convert from (x, y, width, height) to (start_x, start_y, end_x, end_y)
        uint32_t start_x = x_;
        uint32_t start_y = y_;
        uint32_t end_x = x_ + width_ - 1;
        uint32_t end_y = y_ + height_ - 1;

        Metavision::I_DigitalCrop::Region region(start_x, start_y, end_x, end_y);
        digital_crop_->set_window_region(region, false);

        std::cout << "Digital crop region set to [" << start_x << ", " << start_y
                 << ", " << end_x << ", " << end_y << "]" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "DigitalCropFeature: Failed to apply settings: " << e.what() << std::endl;
    }
}

void DigitalCropFeature::set_region(int x, int y, int width, int height) {
    x_ = x;
    y_ = y;
    width_ = width;
    height_ = height;
    apply_settings();
}

bool DigitalCropFeature::render_ui() {
    if (!is_available()) {
        return false;
    }

    bool changed = false;

    if (ImGui::TreeNode("Digital Crop")) {
        ImGui::TextWrapped("%s", description().c_str());
        ImGui::Spacing();

        // Enable/disable checkbox
        bool enabled_ui = enabled_;
        if (ImGui::Checkbox("Enable Digital Crop", &enabled_ui)) {
            enable(enabled_ui);
            changed = true;
        }

        ImGui::Spacing();

        // Position and size controls
        ImGui::Text("Crop Region:");

        int max_width = display_settings_.get_image_width();
        int max_height = display_settings_.get_image_height();

        bool region_changed = false;
        region_changed |= ImGui::SliderInt("X##dc", &x_, 0, max_width - 1);
        region_changed |= ImGui::SliderInt("Y##dc", &y_, 0, max_height - 1);
        region_changed |= ImGui::SliderInt("Width##dc", &width_, 1, max_width);
        region_changed |= ImGui::SliderInt("Height##dc", &height_, 1, max_height);

        if (region_changed && enabled_) {
            apply_settings();
            changed = true;
        }

        ImGui::Spacing();
        ImGui::TextWrapped("Note: Digital crop reduces sensor resolution");
        ImGui::TextWrapped("Region: [%d, %d] to [%d, %d] (%dx%d)",
                          x_, y_, x_ + width_ - 1, y_ + height_ - 1,
                          width_, height_);

        ImGui::TreePop();
    }

    return changed;
}

} // namespace EventCamera
