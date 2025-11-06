#include "core/event_metrics.h"

namespace core {

void EventMetrics::record_events(int64_t count, int64_t timestamp_us) {
    total_events_received_.fetch_add(count);
    events_since_last_update_.fetch_add(count);
    update_rate(timestamp_us);
}

void EventMetrics::record_event_timestamp(int64_t timestamp_us) {
    last_event_camera_ts_.store(timestamp_us);
}

int64_t EventMetrics::get_total_events() const {
    return total_events_received_.load();
}

int64_t EventMetrics::get_events_per_second() const {
    return events_last_second_.load();
}

int64_t EventMetrics::get_last_event_timestamp() const {
    return last_event_camera_ts_.load();
}

void EventMetrics::reset() {
    total_events_received_.store(0);
    events_last_second_.store(0);
    last_event_rate_update_us_.store(0);
    last_event_camera_ts_.store(0);
    events_since_last_update_.store(0);
}

void EventMetrics::update_rate(int64_t timestamp_us) {
    int64_t last_update = last_event_rate_update_us_.load();

    // Update rate every second
    if (timestamp_us - last_update >= 1000000) {  // 1 second
        std::lock_guard<std::mutex> lock(mutex_);

        // Double-check after acquiring lock
        last_update = last_event_rate_update_us_.load();
        if (timestamp_us - last_update >= 1000000) {
            int64_t events = events_since_last_update_.exchange(0);
            events_last_second_.store(events);
            last_event_rate_update_us_.store(timestamp_us);
        }
    }
}

} // namespace core
