#ifndef SETTINGS_PANEL_H
#define SETTINGS_PANEL_H

#include "ui/ui_panel.h"
#include "app_config.h"
#include "camera/bias_manager.h"

// Forward declarations
namespace core {
    class AppState;
}

namespace ui {

/**
 * @brief Settings panel for camera biases and display settings
 *
 * Handles:
 * - Camera connection/reconnection
 * - Bias settings (via BiasManager)
 * - Display settings (FPS, frame subtraction)
 * - Frame generation settings
 */
class SettingsPanel : public UIPanel {
public:
    SettingsPanel(core::AppState& state,
                  AppConfig& config,
                  EventCamera::BiasManager& bias_mgr);

    void render() override;
    std::string title() const override { return "Camera Settings"; }

    bool settings_changed() const { return settings_changed_; }
    void reset_settings_changed() { settings_changed_ = false; }

    bool camera_reconnect_requested() const { return camera_reconnect_requested_; }
    void set_camera_reconnect_request() { camera_reconnect_requested_ = true; }
    void reset_camera_reconnect_request() { camera_reconnect_requested_ = false; }
    bool camera_connect_requested() const { return camera_connect_requested_; }
    void set_camera_connect_request() { camera_connect_requested_ = true; }
    void reset_camera_connect_request() { camera_connect_requested_ = false; }

    // Public render methods for custom layout
    void render_connection_controls();
    void render_bias_controls();
    void render_digital_features();
    void render_display_settings();
    void render_frame_generation();
    void render_genetic_algorithm();
    void render_apply_button();
    void capture_frame();

private:

    core::AppState& state_;
    AppConfig& config_;
    EventCamera::BiasManager& bias_mgr_;

    bool settings_changed_ = false;
    bool camera_reconnect_requested_ = false;
    bool camera_connect_requested_ = false;
    AppConfig::CameraSettings previous_settings_;
};

} // namespace ui

#endif // SETTINGS_PANEL_H
