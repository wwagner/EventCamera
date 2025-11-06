#include "video/filters/subtraction_filter.h"

namespace video {

cv::Mat SubtractionFilter::apply(const cv::Mat& input) {
    if (input.empty()) {
        return input;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (!enabled_) {
        return input;
    }

    cv::Mat output;

    if (!previous_frame_.empty()) {
        // Ensure both frames have the same size and type
        if (input.size() == previous_frame_.size() &&
            input.type() == previous_frame_.type()) {

            // Convert to float for subtraction to avoid underflow
            cv::Mat float_current, float_previous;
            input.convertTo(float_current, CV_32F);
            previous_frame_.convertTo(float_previous, CV_32F);

            // Subtract previous from current
            cv::Mat diff = float_current - float_previous;

            // Scale and shift to make differences visible
            // Add 127.5 to center at gray (so negative diffs are darker, positive are brighter)
            diff = diff + 127.5f;

            // Clamp to valid range [0, 255]
            diff.convertTo(output, CV_8U);
        } else {
            output = input.clone();
        }
    } else {
        output = input.clone();
    }

    // Store current frame as previous for next iteration
    previous_frame_ = output.clone();

    return output;
}

void SubtractionFilter::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    previous_frame_ = cv::Mat();
}

void SubtractionFilter::set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
    if (!enabled_) {
        previous_frame_ = cv::Mat();  // Clear previous frame when disabled
    }
}

bool SubtractionFilter::is_enabled() const {
    return enabled_;
}

std::string SubtractionFilter::name() const {
    return "FrameSubtraction";
}

std::string SubtractionFilter::description() const {
    return "Frame-to-frame subtraction to highlight changes";
}

} // namespace video
