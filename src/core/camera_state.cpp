#include "core/camera_state.h"
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include "camera_manager.h"
#include "core/frame_sync.h"

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

std::unique_ptr<Metavision::PeriodicFrameGenerationAlgorithm>& CameraState::frame_generator(int camera_index) {
    return frame_gens_[camera_index];
}

std::unique_ptr<std::thread>& CameraState::event_thread(int camera_index) {
    return event_threads_[camera_index];
}

int CameraState::num_cameras() const {
    if (!camera_mgr_) return 0;
    return camera_mgr_->num_cameras();
}

std::mutex& CameraState::connection_mutex() {
    return connection_mutex_;
}

FrameSync& CameraState::frame_sync(int camera_index) {
    if (!frame_syncs_[camera_index]) {
        frame_syncs_[camera_index] = std::make_unique<FrameSync>();
    }
    return *frame_syncs_[camera_index];
}

std::mutex& CameraState::frame_gen_mutex(int camera_index) {
    return frame_gen_mutexes_[camera_index];
}

} // namespace core
