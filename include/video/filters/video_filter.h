#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace video {

/**
 * Interface for pluggable video filters
 *
 * All image processing filters implement this interface, allowing them to be
 * chained together in a processing pipeline.
 */
class IVideoFilter {
public:
    virtual ~IVideoFilter() = default;

    /**
     * Apply filter to frame
     * @param input Input frame
     * @return Processed frame (may be the same as input if filter is disabled)
     */
    virtual cv::Mat apply(const cv::Mat& input) = 0;

    /**
     * Enable or disable filter
     * @param enabled true to enable, false to disable
     */
    virtual void set_enabled(bool enabled) = 0;

    /**
     * Check if filter is enabled
     * @return true if enabled
     */
    virtual bool is_enabled() const = 0;

    /**
     * Get filter name
     * @return Short name for this filter
     */
    virtual std::string name() const = 0;

    /**
     * Get filter description
     * @return Human-readable description of what this filter does
     */
    virtual std::string description() const = 0;
};

} // namespace video
