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
    class FrameSync;  // Forward declaration
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
     * Get frame generator for camera index
     * @param camera_index Camera index (0 or 1)
     * @return Reference to frame generator unique_ptr
     */
    std::unique_ptr<Metavision::PeriodicFrameGenerationAlgorithm>& frame_generator(int camera_index = 0);

    /**
     * Get event processing thread for camera index
     * @param camera_index Camera index (0 or 1)
     * @return Reference to event thread unique_ptr
     */
    std::unique_ptr<std::thread>& event_thread(int camera_index = 0);

    /**
     * Get number of active cameras
     */
    int num_cameras() const;

    /**
     * Get connection mutex
     * @return Reference to connection mutex
     */
    std::mutex& connection_mutex();

    /**
     * Get frame sync for camera index
     * @param camera_index Camera index (0 or 1)
     * @return Reference to FrameSync object for this camera
     */
    FrameSync& frame_sync(int camera_index = 0);

    /**
     * Get per-camera frame generator mutex
     *
     * PERFORMANCE: Per-camera mutexes eliminate contention between cameras.
     * Used only for frame generator lifecycle (create/destroy), NOT for
     * event processing (which is lock-free).
     *
     * @param camera_index Camera index (0 or 1)
     * @return Reference to frame generator mutex for this camera
     */
    std::mutex& frame_gen_mutex(int camera_index = 0);

private:
    static constexpr int MAX_CAMERAS = 2;

    std::unique_ptr<CameraManager> camera_mgr_;
    std::unique_ptr<Metavision::PeriodicFrameGenerationAlgorithm> frame_gens_[MAX_CAMERAS];
    std::unique_ptr<std::thread> event_threads_[MAX_CAMERAS];
    std::unique_ptr<FrameSync> frame_syncs_[MAX_CAMERAS];
    std::atomic<bool> camera_connected_{false};
    std::atomic<bool> simulation_mode_{false};
    std::atomic<int64_t> camera_start_time_us_{0};
    std::mutex connection_mutex_;
    std::mutex frame_gen_mutexes_[MAX_CAMERAS];  // Per-camera frame generator protection
};

} // namespace core
