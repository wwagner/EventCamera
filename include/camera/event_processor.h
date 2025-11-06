#ifndef EVENT_PROCESSOR_H
#define EVENT_PROCESSOR_H

#include <metavision/sdk/base/events/event_cd.h>
#include <cstdint>

// Forward declarations
namespace core {
    class AppState;
}

namespace EventCamera {

/**
 * @brief Event processing logic
 *
 * Handles processing of camera event batches, including:
 * - Event metrics tracking
 * - Batch age validation
 * - Frame generation coordination
 */
class EventProcessor {
public:
    EventProcessor(core::AppState& state);
    ~EventProcessor() = default;

    /**
     * @brief Process a batch of events
     * @param begin Iterator to first event
     * @param end Iterator past last event
     */
    void process_events(const Metavision::EventCD* begin,
                       const Metavision::EventCD* end);

    /**
     * @brief Check if event batch is too old
     * @param event_ts Event timestamp in microseconds
     * @return true if batch should be skipped
     */
    bool is_batch_too_old(int64_t event_ts) const;

    /**
     * @brief Update event metrics from batch
     * @param begin Iterator to first event
     * @param end Iterator past last event
     */
    void update_event_metrics(const Metavision::EventCD* begin,
                             const Metavision::EventCD* end);

private:
    core::AppState& state_;
};

} // namespace EventCamera

#endif // EVENT_PROCESSOR_H
