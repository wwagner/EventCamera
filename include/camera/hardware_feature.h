#ifndef HARDWARE_FEATURE_H
#define HARDWARE_FEATURE_H

#include <string>
#include <metavision/sdk/driver/camera.h>

namespace EventCamera {

/**
 * @brief Category of hardware feature for UI organization
 */
enum class FeatureCategory {
    Monitoring,      // Read-only hardware info (sensor temp, illumination, etc.)
    RegionControl,   // ROI, Digital Crop
    EventFiltering,  // ERC, Anti-Flicker, Trail Filter
    Advanced         // Other features
};

/**
 * @brief Interface for all camera hardware features
 *
 * This interface provides a unified way to manage camera features,
 * including lifecycle management, availability checks, settings application,
 * and UI rendering.
 */
class IHardwareFeature {
public:
    virtual ~IHardwareFeature() = default;

    /**
     * @brief Initialize the feature with the camera
     * @param camera Reference to the Metavision camera
     * @return true if initialization succeeded and feature is available
     */
    virtual bool initialize(Metavision::Camera& camera) = 0;

    /**
     * @brief Add an additional camera to be controlled
     * @param camera Reference to the Metavision camera
     * @return true if successful
     */
    virtual bool add_camera(Metavision::Camera& camera) = 0;

    /**
     * @brief Shutdown and cleanup the feature
     */
    virtual void shutdown() = 0;

    /**
     * @brief Check if this feature is available on the camera
     * @return true if the camera hardware supports this feature
     */
    virtual bool is_available() const = 0;

    /**
     * @brief Check if this feature is currently enabled
     * @return true if the feature is enabled
     */
    virtual bool is_enabled() const = 0;

    /**
     * @brief Enable or disable the feature
     * @param enabled true to enable, false to disable
     */
    virtual void enable(bool enabled) = 0;

    /**
     * @brief Apply current settings to the camera hardware
     *
     * Called when settings have changed and need to be pushed to the camera.
     */
    virtual void apply_settings() = 0;

    /**
     * @brief Get the feature name
     * @return Short name for the feature (e.g., "ERC", "ROI")
     */
    virtual std::string name() const = 0;

    /**
     * @brief Get the feature description
     * @return Longer description of what the feature does
     */
    virtual std::string description() const = 0;

    /**
     * @brief Get the feature category
     * @return Category for UI organization
     */
    virtual FeatureCategory category() const = 0;

    /**
     * @brief Render ImGui UI for this feature
     * @return true if any settings were changed
     */
    virtual bool render_ui() = 0;
};

} // namespace EventCamera

#endif // HARDWARE_FEATURE_H
