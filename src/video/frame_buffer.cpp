#include "video/frame_buffer.h"

namespace video {

void FrameBuffer::store_frame(const cv::Mat& frame) {
    if (frame.empty()) {
        return;
    }

    // Only store new frame if previous frame was consumed
    // This prevents frame queue buildup and maintains real-time display
    if (!frame_consumed_.load()) {
        frames_dropped_++;
        return;  // Drop this frame - previous frame not yet displayed
    }

    std::lock_guard<std::mutex> lock(mutex_);
    current_frame_ = frame.clone();
    frame_consumed_.store(false);  // Mark new frame as not yet consumed
    frames_generated_++;
}

std::optional<cv::Mat> FrameBuffer::consume_frame() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (frame_consumed_.load()) {
        return std::nullopt;  // No new frame available
    }

    frame_consumed_.store(true);
    return current_frame_;
}

bool FrameBuffer::has_unconsumed_frame() const {
    return !frame_consumed_.load();
}

int64_t FrameBuffer::get_frames_dropped() const {
    return frames_dropped_.load();
}

int64_t FrameBuffer::get_frames_generated() const {
    return frames_generated_.load();
}

} // namespace video
