#include "core/app_state.h"
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include "camera_manager.h"

namespace core {

AppState::AppState() {
    // Initialize video subsystems for dual cameras
    for (int i = 0; i < MAX_CAMERAS; ++i) {
        frame_buffers_[i] = std::make_unique<video::FrameBuffer>();
        texture_managers_[i] = std::make_unique<video::TextureManager>();
    }

    frame_processor_ = std::make_unique<video::FrameProcessor>();

    // Create and register video filters
    roi_filter_ = std::make_shared<video::ROIFilter>();
    subtraction_filter_ = std::make_shared<video::SubtractionFilter>();
    frame_processor_->add_filter(roi_filter_);
    frame_processor_->add_filter(subtraction_filter_);

    // Initialize core subsystems
    frame_sync_ = std::make_unique<FrameSync>();
    event_metrics_ = std::make_unique<EventMetrics>();
    display_settings_ = std::make_unique<DisplaySettings>();
    camera_state_ = std::make_unique<CameraState>();
    feature_manager_ = std::make_unique<EventCamera::FeatureManager>();
}

AppState::~AppState() = default;

video::FrameBuffer& AppState::frame_buffer(int camera_index) {
    return *frame_buffers_[camera_index];
}

video::FrameProcessor& AppState::frame_processor() {
    return *frame_processor_;
}

video::TextureManager& AppState::texture_manager(int camera_index) {
    return *texture_managers_[camera_index];
}

std::shared_ptr<video::ROIFilter> AppState::roi_filter() {
    return roi_filter_;
}

std::shared_ptr<video::SubtractionFilter> AppState::subtraction_filter() {
    return subtraction_filter_;
}

FrameSync& AppState::frame_sync() {
    return *frame_sync_;
}

EventMetrics& AppState::event_metrics() {
    return *event_metrics_;
}

DisplaySettings& AppState::display_settings() {
    return *display_settings_;
}

CameraState& AppState::camera_state() {
    return *camera_state_;
}

EventCamera::FeatureManager& AppState::feature_manager() {
    return *feature_manager_;
}

bool AppState::is_running() const {
    return running_.load();
}

void AppState::request_shutdown() {
    running_.store(false);
}

void AppState::reset_running_flag() {
    running_.store(true);
}

} // namespace core
