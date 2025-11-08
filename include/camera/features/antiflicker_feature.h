#ifndef ANTIFLICKER_FEATURE_H
#define ANTIFLICKER_FEATURE_H

#include "camera/hardware_feature.h"
#include <metavision/hal/facilities/i_antiflicker_module.h>

namespace EventCamera {

/**
 * @brief Anti-Flicker filter feature
 *
 * Filters out flicker from artificial lighting (50/60Hz)
 */
class AntiFlickerFeature : public IHardwareFeature {
public:
    AntiFlickerFeature() = default;
    ~AntiFlickerFeature() override = default;

    // IHardwareFeature interface
    bool initialize(Metavision::Camera& camera) override;
    bool add_camera(Metavision::Camera& camera) override;
    void shutdown() override;
    bool is_available() const override { return antiflicker_ != nullptr; }
    bool is_enabled() const override { return enabled_; }
    void enable(bool enabled) override;
    void apply_settings() override;

    std::string name() const override { return "Anti-Flicker"; }
    std::string description() const override {
        return "Anti-Flicker Filter - Remove flicker from artificial lighting (50/60Hz)";
    }
    FeatureCategory category() const override { return FeatureCategory::EventFiltering; }

    bool render_ui() override;

    /**
     * @brief Set filter mode
     * @param mode 0 for BAND_STOP, 1 for BAND_PASS
     */
    void set_filtering_mode(int mode);

    /**
     * @brief Set frequency band
     */
    void set_frequency_band(uint32_t low_freq, uint32_t high_freq);

    /**
     * @brief Set duty cycle
     */
    void set_duty_cycle(uint32_t duty_cycle);

private:
    Metavision::I_AntiFlickerModule* antiflicker_ = nullptr;  // Primary camera
    std::vector<Metavision::I_AntiFlickerModule*> all_antiflicker_;  // All cameras to control
    bool enabled_ = false;
    int mode_ = 0;  // 0=BAND_STOP, 1=BAND_PASS
    int low_freq_ = 100;
    int high_freq_ = 150;
    int duty_cycle_ = 50;
};

} // namespace EventCamera

#endif // ANTIFLICKER_FEATURE_H
