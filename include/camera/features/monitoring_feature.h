#ifndef MONITORING_FEATURE_H
#define MONITORING_FEATURE_H

#include "camera/hardware_feature.h"
#include <metavision/hal/facilities/i_monitoring.h>

namespace EventCamera {

/**
 * @brief Hardware monitoring feature (temperature, illumination, pixel dead time)
 *
 * Provides read-only access to camera sensor monitoring information.
 */
class MonitoringFeature : public IHardwareFeature {
public:
    MonitoringFeature() = default;
    ~MonitoringFeature() override = default;

    // IHardwareFeature interface
    bool initialize(Metavision::Camera& camera) override;
    bool add_camera(Metavision::Camera& camera) override;
    void shutdown() override;
    bool is_available() const override;
    bool is_enabled() const override { return is_available(); }  // Always enabled if available
    void enable(bool enabled) override {}  // Read-only feature, can't be disabled
    void apply_settings() override {}  // Read-only feature, no settings to apply

    std::string name() const override { return "Monitoring"; }
    std::string description() const override {
        return "Real-time hardware monitoring (temperature, illumination, pixel dead time)";
    }
    FeatureCategory category() const override { return FeatureCategory::Monitoring; }

    bool render_ui() override;

private:
    /**
     * @brief Check which monitoring capabilities are supported
     */
    void detect_capabilities();

    Metavision::I_Monitoring* monitoring_ = nullptr;  // Primary camera
    std::vector<Metavision::I_Monitoring*> all_monitoring_;  // All cameras

    // Track which capabilities are supported (on primary camera)
    bool has_temperature_ = false;
    bool has_illumination_ = false;
    bool has_dead_time_ = false;
};

} // namespace EventCamera

#endif // MONITORING_FEATURE_H
