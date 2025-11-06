#include "core/camera_state.h"
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include "camera_manager.h"

namespace core {

void CameraState::set_connected(bool connected) {
    camera_connected_.store(connected);
}

bool CameraState::is_connected() const {
    return camera_connected_.load();
}

void CameraState::set_simulation_mode(bool enabled) {
    simulation_mode_.store(enabled);
}

bool CameraState::is_simulation_mode() const {
    return simulation_mode_.load();
}

void CameraState::set_camera_start_time_us(int64_t time_us) {
    camera_start_time_us_.store(time_us);
}

int64_t CameraState::get_camera_start_time_us() const {
    return camera_start_time_us_.load();
}

std::unique_ptr<CameraManager>& CameraState::camera_manager() {
    return camera_mgr_;
}

std::unique_ptr<Metavision::PeriodicFrameGenerationAlgorithm>& CameraState::frame_generator() {
    return frame_gen_;
}

std::unique_ptr<std::thread>& CameraState::event_thread() {
    return event_thread_;
}

std::mutex& CameraState::connection_mutex() {
    return connection_mutex_;
}

} // namespace core
