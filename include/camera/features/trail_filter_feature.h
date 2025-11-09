#ifndef TRAIL_FILTER_FEATURE_H
#define TRAIL_FILTER_FEATURE_H

#include "camera/hardware_feature.h"
#include <metavision/hal/facilities/i_event_trail_filter_module.h>
#include <vector>

namespace EventCamera {

/**
 * @brief Event Trail Filter feature
 *
 * Filters noise from event bursts and rapid flickering
 */
class TrailFilterFeature : public IHardwareFeature {
public:
    TrailFilterFeature() = default;
    ~TrailFilterFeature() override = default;

    // IHardwareFeature interface
    bool initialize(Metavision::Camera& camera) override;
    bool add_camera(Metavision::Camera& camera) override;
    void shutdown() override;
    bool is_available() const override { return trail_filter_ != nullptr; }
    bool is_enabled() const override { return enabled_; }
    void enable(bool enabled) override;
    void apply_settings() override;

    std::string name() const override { return "Trail Filter"; }
    std::string description() const override {
        return "Event Trail Filter - Filter noise from event bursts and rapid flickering";
    }
    FeatureCategory category() const override { return FeatureCategory::EventFiltering; }

    bool render_ui() override;

    /**
     * @brief Set filter type
     */
    void set_type(Metavision::I_EventTrailFilterModule::Type type);

    /**
     * @brief Set threshold delay in microseconds
     */
    void set_threshold(uint32_t threshold_us);

    /**
     * @brief Sync internal state from camera (call after external config changes)
     */
    void sync_from_camera();

private:
    Metavision::I_EventTrailFilterModule* trail_filter_ = nullptr;  // Primary camera for reading capabilities
    std::vector<Metavision::I_EventTrailFilterModule*> all_trail_filters_;  // All cameras to control
    bool enabled_ = false;
    int filter_type_ = 0;  // 0=TRAIL, 1=STC_CUT_TRAIL, 2=STC_KEEP_TRAIL
    int threshold_us_ = 10000;
};

} // namespace EventCamera

#endif // TRAIL_FILTER_FEATURE_H
