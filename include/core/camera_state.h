#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <cstdint>

// Forward declarations
class CameraManager;
namespace Metavision {
    class PeriodicFrameGenerationAlgorithm;
}

namespace core {

/**
 * Camera connection and lifecycle state
 *
 * Manages camera objects, connection status, and related state.
 */
class CameraState {
public:
    CameraState() = default;
    ~CameraState() = default;

    // Non-copyable
    CameraState(const CameraState&) = delete;
    CameraState& operator=(const CameraState&) = delete;

    /**
     * Set connection state
     * @param connected true if camera is connected
     */
    void set_connected(bool connected);

    /**
     * Check if camera is connected
     * @return true if connected
     */
    bool is_connected() const;

    /**
     * Set simulation mode
     * @param enabled true to enable simulation mode
     */
    void set_simulation_mode(bool enabled);

    /**
     * Check if in simulation mode
     * @return true if simulation mode is enabled
     */
    bool is_simulation_mode() const;

    /**
     * Set camera start time
     * @param time_us System time when camera started (microseconds)
     */
    void set_camera_start_time_us(int64_t time_us);

    /**
     * Get camera start time
     * @return System time when camera started (microseconds)
     */
    int64_t get_camera_start_time_us() const;

    /**
     * Get camera manager
     * @return Reference to camera manager unique_ptr
     */
    std::unique_ptr<CameraManager>& camera_manager();

    /**
     * Get frame generator
     * @return Reference to frame generator unique_ptr
     */
    std::unique_ptr<Metavision::PeriodicFrameGenerationAlgorithm>& frame_generator();

    /**
     * Get event processing thread
     * @return Reference to event thread unique_ptr
     */
    std::unique_ptr<std::thread>& event_thread();

    /**
     * Get connection mutex
     * @return Reference to connection mutex
     */
    std::mutex& connection_mutex();

private:
    std::unique_ptr<CameraManager> camera_mgr_;
    std::unique_ptr<Metavision::PeriodicFrameGenerationAlgorithm> frame_gen_;
    std::unique_ptr<std::thread> event_thread_;
    std::atomic<bool> camera_connected_{false};
    std::atomic<bool> simulation_mode_{false};
    std::atomic<int64_t> camera_start_time_us_{0};
    std::mutex connection_mutex_;
};

} // namespace core
