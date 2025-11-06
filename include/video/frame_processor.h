#pragma once

#include "video/filters/video_filter.h"
#include <memory>
#include <vector>
#include <string>
#include <mutex>

namespace video {

/**
 * Frame processor - orchestrates filter pipeline
 *
 * Manages a collection of filters and applies them in sequence to process frames.
 */
class FrameProcessor {
public:
    FrameProcessor() = default;
    ~FrameProcessor() = default;

    // Non-copyable
    FrameProcessor(const FrameProcessor&) = delete;
    FrameProcessor& operator=(const FrameProcessor&) = delete;

    /**
     * Add filter to the end of the pipeline
     * @param filter Filter to add
     */
    void add_filter(std::shared_ptr<IVideoFilter> filter);

    /**
     * Remove filter by name
     * @param name Name of filter to remove
     */
    void remove_filter(const std::string& name);

    /**
     * Get filter by name
     * @param name Name of filter
     * @return Filter if found, nullptr otherwise
     */
    std::shared_ptr<IVideoFilter> get_filter(const std::string& name);

    /**
     * Process frame through all enabled filters
     * @param input Input frame
     * @return Processed frame
     */
    cv::Mat process(const cv::Mat& input);

    /**
     * Convert BGR to RGB (OpenCV uses BGR by default)
     * @param frame BGR frame
     * @return RGB frame
     */
    static cv::Mat bgr_to_rgb(const cv::Mat& frame);

private:
    std::vector<std::shared_ptr<IVideoFilter>> filters_;
    mutable std::mutex mutex_;
};

} // namespace video
