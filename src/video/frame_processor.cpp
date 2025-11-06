#include "video/frame_processor.h"

namespace video {

void FrameProcessor::add_filter(std::shared_ptr<IVideoFilter> filter) {
    if (!filter) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    filters_.push_back(filter);
}

void FrameProcessor::remove_filter(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    filters_.erase(
        std::remove_if(filters_.begin(), filters_.end(),
            [&name](const std::shared_ptr<IVideoFilter>& filter) {
                return filter && filter->name() == name;
            }),
        filters_.end()
    );
}

std::shared_ptr<IVideoFilter> FrameProcessor::get_filter(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& filter : filters_) {
        if (filter && filter->name() == name) {
            return filter;
        }
    }

    return nullptr;
}

cv::Mat FrameProcessor::process(const cv::Mat& input) {
    if (input.empty()) {
        return input;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    cv::Mat result = input;

    // Apply all filters in sequence
    for (auto& filter : filters_) {
        if (filter && filter->is_enabled()) {
            result = filter->apply(result);
        }
    }

    return result;
}

cv::Mat FrameProcessor::bgr_to_rgb(const cv::Mat& frame) {
    if (frame.empty()) {
        return frame;
    }

    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
    return rgb_frame;
}

} // namespace video
