#pragma once

#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>
#include <optional>

namespace video {

/**
 * Thread-safe frame storage with frame dropping
 *
 * Implements a single-frame buffer that drops new frames if the previous
 * frame has not been consumed yet. This prevents frame queue buildup and
 * maintains real-time display.
 */
class FrameBuffer {
public:
    FrameBuffer() = default;
    ~FrameBuffer() = default;

    // Non-copyable
    FrameBuffer(const FrameBuffer&) = delete;
    FrameBuffer& operator=(const FrameBuffer&) = delete;

    /**
     * Store new frame (may drop if not consumed)
     * @param frame Frame to store (will be cloned)
     */
    void store_frame(const cv::Mat& frame);

    /**
     * Consume frame for display
     * @return Frame if available, nullopt otherwise
     */
    std::optional<cv::Mat> consume_frame();

    /**
     * Check if frame is ready
     * @return true if unconsumed frame is available
     */
    bool has_unconsumed_frame() const;

    /**
     * Get number of frames dropped
     * @return Count of frames dropped because buffer was full
     */
    int64_t get_frames_dropped() const;

    /**
     * Get number of frames generated
     * @return Total count of frames stored
     */
    int64_t get_frames_generated() const;

private:
    cv::Mat current_frame_;
    std::atomic<bool> frame_consumed_{true};
    std::atomic<int64_t> frames_dropped_{0};
    std::atomic<int64_t> frames_generated_{0};
    mutable std::mutex mutex_;
};

} // namespace video
