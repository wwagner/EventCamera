#ifndef ERC_FEATURE_H
#define ERC_FEATURE_H

#include "camera/hardware_feature.h"
#include <metavision/hal/facilities/i_erc_module.h>

namespace EventCamera {

/**
 * @brief Event Rate Controller (ERC) feature
 *
 * Limits the maximum event rate to prevent bandwidth saturation.
 */
class ERCFeature : public IHardwareFeature {
public:
    ERCFeature() = default;
    ~ERCFeature() override = default;

    // IHardwareFeature interface
    bool initialize(Metavision::Camera& camera) override;
    void shutdown() override;
    bool is_available() const override { return erc_ != nullptr; }
    bool is_enabled() const override { return enabled_; }
    void enable(bool enabled) override;
    void apply_settings() override;

    std::string name() const override { return "ERC"; }
    std::string description() const override {
        return "Event Rate Controller - Limit maximum event rate to prevent bandwidth saturation";
    }
    FeatureCategory category() const override { return FeatureCategory::EventFiltering; }

    bool render_ui() override;

    /**
     * @brief Set target event rate in kilo-events per second
     */
    void set_event_rate_kevps(int rate_kevps);

    /**
     * @brief Get current event rate setting
     */
    int get_event_rate_kevps() const { return target_rate_kevps_; }

private:
    Metavision::I_ErcModule* erc_ = nullptr;
    bool enabled_ = false;
    int target_rate_kevps_ = 1000;  // kilo-events per second
};

} // namespace EventCamera

#endif // ERC_FEATURE_H
