#ifndef DIGITAL_CROP_FEATURE_H
#define DIGITAL_CROP_FEATURE_H

#include "camera/hardware_feature.h"
#include "core/display_settings.h"
#include <metavision/hal/facilities/i_digital_crop.h>

namespace EventCamera {

/**
 * @brief Digital Crop hardware feature
 *
 * Crops sensor output to reduce resolution and data volume
 */
class DigitalCropFeature : public IHardwareFeature {
public:
    DigitalCropFeature(core::DisplaySettings& display_settings);
    ~DigitalCropFeature() override = default;

    // IHardwareFeature interface
    bool initialize(Metavision::Camera& camera) override;
    void shutdown() override;
    bool is_available() const override { return digital_crop_ != nullptr; }
    bool is_enabled() const override { return enabled_; }
    void enable(bool enabled) override;
    void apply_settings() override;

    std::string name() const override { return "Digital Crop"; }
    std::string description() const override {
        return "Digital Crop - Crop sensor output to reduce resolution and data volume";
    }
    FeatureCategory category() const override { return FeatureCategory::RegionControl; }

    bool render_ui() override;

    /**
     * @brief Set crop region
     */
    void set_region(int x, int y, int width, int height);

private:
    Metavision::I_DigitalCrop* digital_crop_ = nullptr;
    core::DisplaySettings& display_settings_;

    bool enabled_ = false;
    int x_ = 0;
    int y_ = 0;
    int width_ = 640;
    int height_ = 480;
};

} // namespace EventCamera

#endif // DIGITAL_CROP_FEATURE_H
