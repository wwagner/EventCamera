#pragma once

#include "video/filters/video_filter.h"
#include <mutex>

namespace video {

/**
 * Region of Interest (ROI) filter
 *
 * Can either draw a rectangle around the ROI or crop the frame to show
 * only the ROI region.
 */
class ROIFilter : public IVideoFilter {
public:
    ROIFilter() = default;
    ~ROIFilter() override = default;

    /**
     * Set ROI region
     * @param x X coordinate of top-left corner
     * @param y Y coordinate of top-left corner
     * @param width Width of ROI
     * @param height Height of ROI
     */
    void set_roi(int x, int y, int width, int height);

    /**
     * Set whether to crop to ROI or just draw rectangle
     * @param crop true to crop, false to draw rectangle
     */
    void set_crop_to_roi(bool crop);

    /**
     * Set whether to show ROI rectangle (when not cropping)
     * @param show true to show rectangle
     */
    void set_show_rectangle(bool show);

    // IVideoFilter interface
    cv::Mat apply(const cv::Mat& input) override;
    void set_enabled(bool enabled) override;
    bool is_enabled() const override;
    std::string name() const override;
    std::string description() const override;

private:
    bool enabled_ = false;
    bool crop_to_roi_ = false;
    bool show_rectangle_ = false;
    int x_ = 0;
    int y_ = 0;
    int width_ = 640;
    int height_ = 360;
    mutable std::mutex mutex_;
};

} // namespace video
