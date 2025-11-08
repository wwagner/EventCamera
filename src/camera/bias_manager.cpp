#include "camera/bias_manager.h"
#include <imgui.h>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace EventCamera {

bool BiasManager::initialize(Metavision::Camera& camera) {
    std::lock_guard<std::mutex> lock(mutex_);

    ll_biases_ = camera.get_device().get_facility<Metavision::I_LL_Biases>();
    if (!ll_biases_) {
        std::cerr << "BiasManager: Camera does not support bias control" << std::endl;
        return false;
    }

    // Add to list of all cameras to control
    all_ll_biases_.clear();
    all_ll_biases_.push_back(ll_biases_);

    // First, try to list ALL available biases from the camera
    std::cout << "\n=== BiasManager: Querying all available biases from camera ===" << std::endl;

    // Query ranges for biases we know about
    std::vector<std::string> bias_names = {"bias_diff", "bias_diff_on", "bias_diff_off", "bias_fo", "bias_hpf", "bias_refr"};

    bias_ranges_.clear();
    for (const auto& name : bias_names) {
        try {
            Metavision::LL_Bias_Info info;
            if (ll_biases_->get_bias_info(name, info)) {
                auto range = info.get_bias_range();
                int current = ll_biases_->get(name);
                bias_ranges_[name] = {range.first, range.second, current};

                // Initialize slider position
                slider_positions_[name] = bias_to_slider(current, range.first, range.second);

                std::cout << "  ✓ " << name << ": range=[" << range.first
                         << " to " << range.second << "], current=" << current << std::endl;
            } else {
                std::cout << "  ✗ " << name << ": NOT SUPPORTED by hardware" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "  ✗ " << name << ": ERROR - " << e.what() << std::endl;
        }
    }

    std::cout << "BiasManager: Initialized with " << bias_ranges_.size() << " supported biases" << std::endl;
    std::cout << "============================================================\n" << std::endl;
    return !bias_ranges_.empty();
}

bool BiasManager::add_camera(Metavision::Camera& camera) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto* camera_biases = camera.get_device().get_facility<Metavision::I_LL_Biases>();
    if (!camera_biases) {
        std::cerr << "BiasManager: Additional camera does not support bias control" << std::endl;
        return false;
    }

    all_ll_biases_.push_back(camera_biases);
    std::cout << "BiasManager: Added camera (now controlling " << all_ll_biases_.size() << " cameras)" << std::endl;
    return true;
}

void BiasManager::setup_simulation_defaults() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Set up default ranges for simulation mode (matching hardware specs)
    bias_ranges_["bias_diff"] = {-25, 23, 0};
    bias_ranges_["bias_diff_on"] = {-85, 140, 0};
    bias_ranges_["bias_diff_off"] = {-35, 190, 0};
    bias_ranges_["bias_fo"] = {-35, 55, 0};
    bias_ranges_["bias_hpf"] = {0, 120, 0};
    bias_ranges_["bias_refr"] = {-20, 235, 0};

    // Initialize slider positions to middle
    for (const auto& [name, range] : bias_ranges_) {
        slider_positions_[name] = bias_to_slider(range.current, range.min, range.max);
    }

    std::cout << "BiasManager: Set up simulation defaults" << std::endl;
}

bool BiasManager::set_bias(const std::string& name, int value) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = bias_ranges_.find(name);
    if (it == bias_ranges_.end()) {
        std::cerr << "BiasManager: Bias '" << name << "' not found" << std::endl;
        return false;
    }

    // Clamp to range
    value = std::clamp(value, it->second.min, it->second.max);
    it->second.current = value;

    return true;
}

int BiasManager::get_bias(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = bias_ranges_.find(name);
    if (it != bias_ranges_.end()) {
        return it->second.current;
    }

    return 0;
}

void BiasManager::apply_to_camera() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (all_ll_biases_.empty()) {
        std::cout << "BiasManager: No cameras connected, skipping bias application" << std::endl;
        return;
    }

    std::cout << "BiasManager: Applying biases to " << all_ll_biases_.size() << " camera(s)..." << std::endl;

    // Apply to all cameras
    for (size_t cam_idx = 0; cam_idx < all_ll_biases_.size(); ++cam_idx) {
        auto* camera_biases = all_ll_biases_[cam_idx];
        if (!camera_biases) continue;

        std::cout << "  Camera " << cam_idx << ":" << std::endl;
        for (const auto& [name, range] : bias_ranges_) {
            try {
                camera_biases->set(name, range.current);
                std::cout << "    " << name << "=" << range.current << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "    Warning: Could not set " << name << ": " << e.what() << std::endl;
            }
        }
    }

    std::cout << "BiasManager: Biases applied successfully to all cameras" << std::endl;
}

void BiasManager::reset_to_defaults() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::cout << "BiasManager: Resetting biases to defaults (middle of range)" << std::endl;

    for (auto& [name, range] : bias_ranges_) {
        range.current = (range.min + range.max) / 2;
        slider_positions_[name] = bias_to_slider(range.current, range.min, range.max);
    }
}

bool BiasManager::render_bias_ui(const std::string& name, const std::string& label,
                                 const std::string& description) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = bias_ranges_.find(name);
    if (it == bias_ranges_.end()) {
        return false;
    }

    auto& range = it->second;
    bool changed = false;

    // Get or initialize slider position
    if (slider_positions_.find(name) == slider_positions_.end()) {
        slider_positions_[name] = bias_to_slider(range.current, range.min, range.max);
    }

    float& slider_pos = slider_positions_[name];

    // Render slider with actual bias value
    std::string slider_label = label.empty() ? name : label;
    int temp_value = range.current;
    if (ImGui::SliderInt((slider_label + "##" + name).c_str(), &temp_value,
                         range.min, range.max)) {
        range.current = temp_value;
        slider_pos = bias_to_slider(temp_value, range.min, range.max);
        changed = true;
    }

    // Show description and range
    if (!description.empty()) {
        ImGui::TextWrapped("%s [%d to %d]",
                          description.c_str(), range.min, range.max);
        ImGui::Spacing();
    }

    return changed;
}

int BiasManager::slider_to_bias(float slider_pct, int min, int max) const {
    // Exponential mapping for better control at low values
    float normalized = slider_pct / 100.0f;  // 0.0 to 1.0
    float exponential = std::pow(normalized, 2.5f);  // More resolution at low end
    return static_cast<int>(min + exponential * (max - min));
}

float BiasManager::bias_to_slider(int bias, int min, int max) const {
    // Inverse exponential mapping
    float normalized = (bias - min) / static_cast<float>(max - min);
    return std::pow(normalized, 1.0f / 2.5f) * 100.0f;
}

} // namespace EventCamera
