#include "core/frame_sync.h"

namespace core {

void FrameSync::on_frame_generated(int64_t camera_ts, int64_t system_ts) {
    last_frame_camera_ts_.store(camera_ts);
    last_frame_system_ts_.store(system_ts);
}

void FrameSync::on_frame_displayed(int64_t system_ts) {
    last_display_time_us_.store(system_ts);
}

int64_t FrameSync::get_last_frame_camera_ts() const {
    return last_frame_camera_ts_.load();
}

int64_t FrameSync::get_last_frame_system_ts() const {
    return last_frame_system_ts_.load();
}

int64_t FrameSync::get_last_display_time_us() const {
    return last_display_time_us_.load();
}

bool FrameSync::should_display_frame(int64_t current_time_us, int target_fps) const {
    int64_t last_display = last_display_time_us_.load();
    int64_t frame_interval_us = 1000000 / target_fps;  // microseconds between frames
    return (current_time_us - last_display) >= frame_interval_us;
}

} // namespace core
