#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include "hardware_feature.h"
#include <vector>
#include <memory>
#include <mutex>
#include <string>

namespace EventCamera {

/**
 * @brief Manages a collection of hardware features
 *
 * The FeatureManager handles registration, initialization, and lifecycle
 * management of all camera hardware features.
 */
class FeatureManager {
public:
    FeatureManager() = default;
    ~FeatureManager() = default;

    /**
     * @brief Register a feature with the manager
     * @param feature Shared pointer to the feature to register
     */
    void register_feature(std::shared_ptr<IHardwareFeature> feature);

    /**
     * @brief Initialize all registered features with the camera
     * @param camera Reference to the Metavision camera
     *
     * Calls initialize() on all registered features. Features that fail
     * initialization or are not available will be marked as unavailable.
     */
    void initialize_all(Metavision::Camera& camera);

    /**
     * @brief Add an additional camera to be controlled by all features
     * @param camera Reference to the Metavision camera
     * @return true if successful
     */
    bool add_camera(Metavision::Camera& camera);

    /**
     * @brief Shutdown all features
     */
    void shutdown_all();

    /**
     * @brief Clear all registered features
     *
     * Removes all features from the manager. Should be called after shutdown_all().
     */
    void clear();

    /**
     * @brief Get all features in a specific category
     * @param category The feature category to filter by
     * @return Vector of features in the specified category
     */
    std::vector<std::shared_ptr<IHardwareFeature>>
        get_features_by_category(FeatureCategory category);

    /**
     * @brief Get a specific feature by name
     * @param name Name of the feature to retrieve
     * @return Shared pointer to the feature, or nullptr if not found
     */
    std::shared_ptr<IHardwareFeature> get_feature(const std::string& name);

    /**
     * @brief Render UI for all available features
     *
     * Renders ImGui UI for all features that are available on the camera.
     * Features are organized by category.
     */
    void render_all_ui();

    /**
     * @brief Apply settings for all enabled features
     */
    void apply_all_settings();

    /**
     * @brief Get all registered features
     * @return Vector of all registered features
     */
    const std::vector<std::shared_ptr<IHardwareFeature>>& get_all_features() const {
        return features_;
    }

private:
    std::vector<std::shared_ptr<IHardwareFeature>> features_;
    std::mutex mutex_;
};

} // namespace EventCamera

#endif // FEATURE_MANAGER_H
