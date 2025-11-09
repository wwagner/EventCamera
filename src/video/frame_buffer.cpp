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
    // ZERO-COPY: Create FrameRef from cv::Mat (shares data, no clone!)
    current_frame_ = FrameRef(frame);
    frame_consumed_.store(false);  // Mark new frame as not yet consumed
    frames_generated_++;
}

void FrameBuffer::store_frame(const FrameRef& frame_ref) {
    if (frame_ref.empty()) {
        return;
    }

    // Only store new frame if previous frame was consumed
    if (!frame_consumed_.load()) {
        frames_dropped_++;
        return;  // Drop this frame - previous frame not yet displayed
    }

    std::lock_guard<std::mutex> lock(mutex_);
    // ZERO-COPY: Share FrameRef (reference counting, no clone!)
    current_frame_ = frame_ref;
    frame_consumed_.store(false);
    frames_generated_++;
}

void FrameBuffer::store_frame(FrameRef&& frame_ref) {
    if (frame_ref.empty()) {
        return;
    }

    // Only store new frame if previous frame was consumed
    if (!frame_consumed_.load()) {
        frames_dropped_++;
        return;  // Drop this frame - previous frame not yet displayed
    }

    std::lock_guard<std::mutex> lock(mutex_);
    // ZERO-COPY: Move FrameRef (no allocation, just pointer swap)
    current_frame_ = std::move(frame_ref);
    frame_consumed_.store(false);
    frames_generated_++;
}

std::optional<FrameRef> FrameBuffer::consume_frame() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (frame_consumed_.load()) {
        return std::nullopt;  // No new frame available
    }

    frame_consumed_.store(true);
    // ZERO-COPY: Return FrameRef (shares data with stored frame)
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
