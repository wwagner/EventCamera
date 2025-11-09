#pragma once

#include "video/filters/video_filter.h"
#include "video/frame_ref.h"
#include <mutex>

namespace video {

/**
 * Frame subtraction filter (ZERO-COPY optimized)
 *
 * Subtracts the previous frame from the current frame to highlight changes.
 * Useful for motion detection and visualizing temporal differences.
 *
 * PERFORMANCE: Eliminates 3 clone() calls per frame using FrameRef.
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
    FrameRef previous_frame_;  // ZERO-COPY: Use FrameRef instead of cv::Mat
    mutable std::mutex mutex_;
};

} // namespace video
