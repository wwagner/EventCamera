#pragma once

#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>
#include <optional>
#include "video/frame_ref.h"

namespace video {

/**
 * Thread-safe frame storage with frame dropping (ZERO-COPY optimized)
 *
 * Implements a single-frame buffer that drops new frames if the previous
 * frame has not been consumed yet. This prevents frame queue buildup and
 * maintains real-time display.
 *
 * **PERFORMANCE:** Uses FrameRef for zero-copy frame storage.
 * No clone() calls - frames shared via copy-on-write semantics.
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
     * @param frame Frame to store (ZERO-COPY - uses move/share semantics)
     */
    void store_frame(const cv::Mat& frame);

    /**
     * Store new frame from FrameRef (zero-copy share)
     * @param frame_ref Frame reference to share
     */
    void store_frame(const FrameRef& frame_ref);

    /**
     * Store new frame with move semantics (most efficient)
     * @param frame_ref Frame reference to move (zero copy)
     */
    void store_frame(FrameRef&& frame_ref);

    /**
     * Consume frame for display (ZERO-COPY)
     * @return FrameRef if available, nullopt otherwise
     */
    std::optional<FrameRef> consume_frame();

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
    FrameRef current_frame_;
    std::atomic<bool> frame_consumed_{true};
    std::atomic<int64_t> frames_dropped_{0};
    std::atomic<int64_t> frames_generated_{0};
    mutable std::mutex mutex_;
};

} // namespace video
