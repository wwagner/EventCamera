#pragma once

#include "core/frame_sync.h"
#include "core/event_metrics.h"
#include "core/display_settings.h"
#include "core/camera_state.h"
#include "video/frame_buffer.h"
#include "video/frame_processor.h"
#include "video/texture_manager.h"
#include "video/filters/roi_filter.h"
#include "video/filters/subtraction_filter.h"
#include "camera/feature_manager.h"

#include <memory>
#include <atomic>

namespace core {

/**
 * Top-level application state container
 *
 * Provides centralized access to all application subsystems.
 * Replaces global state variables with a clean, thread-safe interface.
 */
class AppState {
public:
    AppState();
    ~AppState();

    // Non-copyable
    AppState(const AppState&) = delete;
    AppState& operator=(const AppState&) = delete;

    // === Subsystem Access ===

    /**
     * Get frame buffer
     * @return Reference to frame buffer
     */
    video::FrameBuffer& frame_buffer();

    /**
     * Get frame processor
     * @return Reference to frame processor
     */
    video::FrameProcessor& frame_processor();

    /**
     * Get texture manager
     * @return Reference to texture manager
     */
    video::TextureManager& texture_manager();

    /**
     * Get ROI filter
     * @return Shared pointer to ROI filter
     */
    std::shared_ptr<video::ROIFilter> roi_filter();

    /**
     * Get subtraction filter
     * @return Shared pointer to subtraction filter
     */
    std::shared_ptr<video::SubtractionFilter> subtraction_filter();

    /**
     * Get frame sync
     * @return Reference to frame sync
     */
    FrameSync& frame_sync();

    /**
     * Get event metrics
     * @return Reference to event metrics
     */
    EventMetrics& event_metrics();

    /**
     * Get display settings
     * @return Reference to display settings
     */
    DisplaySettings& display_settings();

    /**
     * Get camera state
     * @return Reference to camera state
     */
    CameraState& camera_state();

    /**
     * Get feature manager
     * @return Reference to feature manager
     */
    EventCamera::FeatureManager& feature_manager();

    // === Running State ===

    /**
     * Check if application is running
     * @return true if running
     */
    bool is_running() const;

    /**
     * Request application shutdown
     */
    void request_shutdown();

    /**
     * Reset running flag (for camera reconnection)
     */
    void reset_running_flag();

private:
    std::atomic<bool> running_{true};

    // Subsystem instances
    std::unique_ptr<video::FrameBuffer> frame_buffer_;
    std::unique_ptr<video::FrameProcessor> frame_processor_;
    std::unique_ptr<video::TextureManager> texture_manager_;
    std::shared_ptr<video::ROIFilter> roi_filter_;
    std::shared_ptr<video::SubtractionFilter> subtraction_filter_;
    std::unique_ptr<FrameSync> frame_sync_;
    std::unique_ptr<EventMetrics> event_metrics_;
    std::unique_ptr<DisplaySettings> display_settings_;
    std::unique_ptr<CameraState> camera_state_;
    std::unique_ptr<EventCamera::FeatureManager> feature_manager_;
};

} // namespace core
