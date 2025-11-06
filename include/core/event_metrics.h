#pragma once

#include <atomic>
#include <mutex>
#include <cstdint>

namespace core {

/**
 * Event rate tracking and statistics
 *
 * Tracks event counts and calculates event rates.
 */
class EventMetrics {
public:
    EventMetrics() = default;
    ~EventMetrics() = default;

    // Non-copyable
    EventMetrics(const EventMetrics&) = delete;
    EventMetrics& operator=(const EventMetrics&) = delete;

    /**
     * Record events received
     * @param count Number of events
     * @param timestamp_us System timestamp in microseconds
     */
    void record_events(int64_t count, int64_t timestamp_us);

    /**
     * Record event timestamp (for latency tracking)
     * @param timestamp_us Event timestamp from camera
     */
    void record_event_timestamp(int64_t timestamp_us);

    /**
     * Get total events received
     * @return Total event count
     */
    int64_t get_total_events() const;

    /**
     * Get events per second
     * @return Event rate (events/sec)
     */
    int64_t get_events_per_second() const;

    /**
     * Get last event timestamp
     * @return Last event timestamp from camera (microseconds)
     */
    int64_t get_last_event_timestamp() const;

    /**
     * Reset all counters
     */
    void reset();

private:
    void update_rate(int64_t timestamp_us);

    std::atomic<int64_t> total_events_received_{0};
    std::atomic<int64_t> events_last_second_{0};
    std::atomic<int64_t> last_event_rate_update_us_{0};
    std::atomic<int64_t> last_event_camera_ts_{0};
    std::atomic<int64_t> events_since_last_update_{0};
    mutable std::mutex mutex_;
};

} // namespace core
