#include "camera/feature_manager.h"
#include <algorithm>
#include <iostream>

namespace EventCamera {

void FeatureManager::register_feature(std::shared_ptr<IHardwareFeature> feature) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (feature) {
        features_.push_back(feature);
        std::cout << "Registered feature: " << feature->name() << std::endl;
    }
}

void FeatureManager::initialize_all(Metavision::Camera& camera) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "Initializing " << features_.size() << " features..." << std::endl;

    for (auto& feature : features_) {
        try {
            bool success = feature->initialize(camera);
            if (success) {
                std::cout << "  ✓ " << feature->name() << " - available" << std::endl;
            } else {
                std::cout << "  ✗ " << feature->name() << " - not available" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "  ✗ " << feature->name() << " - error: " << e.what() << std::endl;
        }
    }
}

void FeatureManager::shutdown_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "Shutting down features..." << std::endl;

    for (auto& feature : features_) {
        try {
            feature->shutdown();
        } catch (const std::exception& e) {
            std::cerr << "Error shutting down feature " << feature->name()
                     << ": " << e.what() << std::endl;
        }
    }
}

std::vector<std::shared_ptr<IHardwareFeature>>
FeatureManager::get_features_by_category(FeatureCategory category) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<IHardwareFeature>> result;

    for (auto& feature : features_) {
        if (feature->category() == category) {
            result.push_back(feature);
        }
    }

    return result;
}

std::shared_ptr<IHardwareFeature> FeatureManager::get_feature(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = std::find_if(features_.begin(), features_.end(),
        [&name](const std::shared_ptr<IHardwareFeature>& feature) {
            return feature->name() == name;
        });

    if (it != features_.end()) {
        return *it;
    }

    return nullptr;
}

void FeatureManager::render_all_ui() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Render features organized by category
    for (auto& feature : features_) {
        if (feature->is_available()) {
            try {
                feature->render_ui();
            } catch (const std::exception& e) {
                std::cerr << "Error rendering UI for feature " << feature->name()
                         << ": " << e.what() << std::endl;
            }
        }
    }
}

void FeatureManager::apply_all_settings() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& feature : features_) {
        if (feature->is_available() && feature->is_enabled()) {
            try {
                feature->apply_settings();
            } catch (const std::exception& e) {
                std::cerr << "Error applying settings for feature " << feature->name()
                         << ": " << e.what() << std::endl;
            }
        }
    }
}

} // namespace EventCamera
