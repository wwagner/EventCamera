#include "camera/event_processor.h"
#include "core/app_state.h"
#include <iostream>
#include <chrono>

namespace EventCamera {

EventProcessor::EventProcessor(core::AppState& state)
    : state_(state) {
}

void EventProcessor::process_events(const Metavision::EventCD* begin,
                                    const Metavision::EventCD* end) {
    if (begin == end) {
        return;  // No events to process
    }

    // Update event metrics
    update_event_metrics(begin, end);

    // Check if events are too old
    int64_t last_event_ts = (end - 1)->t;
    if (is_batch_too_old(last_event_ts)) {
        return;  // Skip old events
    }

    // Events are processed by the frame generator which is set up separately
    // This is just the metrics and validation logic
}

bool EventProcessor::is_batch_too_old(int64_t event_ts) const {
    // Get current system time
    auto now = std::chrono::steady_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();

    // Convert event timestamp to system time
    int64_t cam_start = state_.camera_state().get_camera_start_time_us();
    if (cam_start == 0) {
        return false;  // Can't determine age without start time
    }

    int64_t event_system_ts = cam_start + event_ts;
    int64_t age_us = now_us - event_system_ts;

    // Consider events older than 5 seconds as too old
    const int64_t MAX_EVENT_AGE_US = 5000000;  // 5 seconds
    if (age_us > MAX_EVENT_AGE_US) {
        std::cerr << "Warning: Skipping old event batch (age: "
                 << (age_us / 1000000.0) << " seconds)" << std::endl;
        return true;
    }

    return false;
}

void EventProcessor::update_event_metrics(const Metavision::EventCD* begin,
                                          const Metavision::EventCD* end) {
    if (begin == end) {
        return;
    }

    // Count events in batch
    int64_t event_count = end - begin;

    // Get timestamp of last event
    int64_t last_event_ts = (end - 1)->t;

    // Update metrics
    state_.event_metrics().record_events(event_count, last_event_ts);
    state_.event_metrics().record_event_timestamp(last_event_ts);
}

} // namespace EventCamera
