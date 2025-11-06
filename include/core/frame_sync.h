#pragma once

#include <atomic>
#include <cstdint>

namespace core {

/**
 * Frame timing and synchronization
 *
 * Tracks frame timestamps and provides rate limiting for display updates.
 */
class FrameSync {
public:
    FrameSync() = default;
    ~FrameSync() = default;

    // Non-copyable
    FrameSync(const FrameSync&) = delete;
    FrameSync& operator=(const FrameSync&) = delete;

    /**
     * Update timestamps when frame is generated
     * @param camera_ts Camera timestamp (microseconds since camera start)
     * @param system_ts System timestamp (microseconds)
     */
    void on_frame_generated(int64_t camera_ts, int64_t system_ts);

    /**
     * Update timestamp when frame is displayed
     * @param system_ts System timestamp (microseconds)
     */
    void on_frame_displayed(int64_t system_ts);

    /**
     * Get last frame camera timestamp
     * @return Camera timestamp in microseconds
     */
    int64_t get_last_frame_camera_ts() const;

    /**
     * Get last frame system timestamp
     * @return System timestamp in microseconds
     */
    int64_t get_last_frame_system_ts() const;

    /**
     * Get last display time
     * @return System timestamp when last frame was displayed (microseconds)
     */
    int64_t get_last_display_time_us() const;

    /**
     * Check if enough time has passed to display next frame
     * @param current_time_us Current system time in microseconds
     * @param target_fps Target display rate
     * @return true if frame should be displayed
     */
    bool should_display_frame(int64_t current_time_us, int target_fps) const;

private:
    std::atomic<int64_t> last_frame_camera_ts_{0};
    std::atomic<int64_t> last_frame_system_ts_{0};
    std::atomic<int64_t> last_display_time_us_{0};
};

} // namespace core
