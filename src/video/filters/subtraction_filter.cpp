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
        // Get read-only access to previous frame (zero-copy)
        ReadGuard prev_guard(previous_frame_);
        const cv::Mat& prev_mat = prev_guard.get();

        // Ensure both frames have the same size and type
        if (input.size() == prev_mat.size() &&
            input.type() == prev_mat.type()) {

            // Convert to float for subtraction to avoid underflow
            cv::Mat float_current, float_previous;
            input.convertTo(float_current, CV_32F);
            prev_mat.convertTo(float_previous, CV_32F);

            // Subtract previous from current
            cv::Mat diff = float_current - float_previous;

            // Scale and shift to make differences visible
            // Add 127.5 to center at gray (so negative diffs are darker, positive are brighter)
            diff = diff + 127.5f;

            // Clamp to valid range [0, 255]
            diff.convertTo(output, CV_8U);
        } else {
            // Size mismatch - just return input as-is
            output = input.clone();
        }
    } else {
        // No previous frame - return input as-is
        output = input.clone();
    }

    // ZERO-COPY: Store output as previous frame (move semantics)
    previous_frame_ = FrameRef(std::move(output));

    // Return a copy for the caller (they expect to own it)
    return previous_frame_.unsafe_get().clone();
}

void SubtractionFilter::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    previous_frame_.reset();
}

void SubtractionFilter::set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
    if (!enabled_) {
        previous_frame_.reset();  // Clear previous frame when disabled
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
