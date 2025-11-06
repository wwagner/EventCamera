#pragma once

#include "video/filters/video_filter.h"
#include <mutex>

namespace video {

/**
 * Frame subtraction filter
 *
 * Subtracts the previous frame from the current frame to highlight changes.
 * Useful for motion detection and visualizing temporal differences.
 */
class SubtractionFilter : public IVideoFilter {
public:
    SubtractionFilter() = default;
    ~SubtractionFilter() override = default;

    /**
     * Reset the filter (clear previous frame)
     */
    void reset();

    // IVideoFilter interface
    cv::Mat apply(const cv::Mat& input) override;
    void set_enabled(bool enabled) override;
    bool is_enabled() const override;
    std::string name() const override;
    std::string description() const override;

private:
    bool enabled_ = false;
    cv::Mat previous_frame_;
    mutable std::mutex mutex_;
};

} // namespace video
