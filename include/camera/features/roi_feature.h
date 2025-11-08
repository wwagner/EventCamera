#ifndef ROI_FEATURE_H
#define ROI_FEATURE_H

#include "camera/hardware_feature.h"
#include "core/display_settings.h"
#include <metavision/hal/facilities/i_roi.h>
#include <memory>

// Forward declaration
namespace video {
    class ROIFilter;
}

namespace EventCamera {

/**
 * @brief Region of Interest (ROI) hardware feature
 *
 * Defines a rectangular region to process or ignore events.
 * Coordinates with the ROIFilter for visualization.
 */
class ROIFeature : public IHardwareFeature {
public:
    ROIFeature(std::shared_ptr<video::ROIFilter> roi_filter, core::DisplaySettings& display_settings);
    ~ROIFeature() override = default;

    // IHardwareFeature interface
    bool initialize(Metavision::Camera& camera) override;
    bool add_camera(Metavision::Camera& camera) override;
    void shutdown() override;
    bool is_available() const override { return roi_ != nullptr; }
    bool is_enabled() const override { return enabled_; }
    void enable(bool enabled) override;
    void apply_settings() override;

    std::string name() const override { return "ROI"; }
    std::string description() const override {
        return "Region of Interest - Define a rectangular region to process or ignore events";
    }
    FeatureCategory category() const override { return FeatureCategory::RegionControl; }

    bool render_ui() override;

    /**
     * @brief Set ROI mode
     */
    void set_mode(Metavision::I_ROI::Mode mode);

    /**
     * @brief Set ROI window
     */
    void set_window(int x, int y, int width, int height);

    /**
     * @brief Set whether to crop the view to ROI
     */
    void set_crop_view(bool crop) { crop_view_ = crop; }

private:
    Metavision::I_ROI* roi_ = nullptr;  // Primary camera
    std::vector<Metavision::I_ROI*> all_roi_;  // All cameras to control
    std::shared_ptr<video::ROIFilter> roi_filter_;
    core::DisplaySettings& display_settings_;

    bool enabled_ = false;
    bool crop_view_ = false;
    int mode_ = 0;  // 0=ROI (keep inside), 1=RONI (discard inside)
    int x_ = 0;
    int y_ = 0;
    int width_ = 320;
    int height_ = 180;
};

} // namespace EventCamera

#endif // ROI_FEATURE_H
