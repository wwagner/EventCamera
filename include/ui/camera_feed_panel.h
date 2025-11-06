#ifndef CAMERA_FEED_PANEL_H
#define CAMERA_FEED_PANEL_H

#include "ui/ui_panel.h"

// Forward declarations
namespace core {
    class AppState;
}

namespace ui {

/**
 * @brief Camera feed panel showing live video and statistics
 *
 * Displays:
 * - Live camera feed with proper aspect ratio
 * - Event rate statistics
 * - Frame generation/drop statistics
 * - Event latency information
 * - Resolution and FPS
 */
class CameraFeedPanel : public UIPanel {
public:
    CameraFeedPanel(core::AppState& state);

    void render() override;
    std::string title() const override { return "Camera Feed"; }

private:
    void render_statistics();
    void render_feed_texture();
    void capture_frame();

    core::AppState& state_;
};

} // namespace ui

#endif // CAMERA_FEED_PANEL_H
