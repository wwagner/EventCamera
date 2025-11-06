#ifndef CAMERA_CONTROLLER_H
#define CAMERA_CONTROLLER_H

#include "camera/bias_manager.h"
#include "camera/feature_manager.h"
#include <memory>
#include <thread>

// Forward declarations
namespace core {
    class AppState;
}
class AppConfig;

namespace EventCamera {

/**
 * @brief High-level camera controller
 *
 * Manages camera lifecycle, event processing, and feature/bias management.
 * Provides a unified interface for camera operations.
 */
class CameraController {
public:
    CameraController(core::AppState& state, AppConfig& config);
    ~CameraController();

    /**
     * @brief Attempt to connect to a camera
     * @param serial_hint Optional camera serial number
     * @return true if connection successful
     */
    bool connect_camera(const std::string& serial_hint = "");

    /**
     * @brief Disconnect from camera
     */
    void disconnect_camera();

    /**
     * @brief Check if camera is connected
     */
    bool is_connected() const;

    /**
     * @brief Start the event processing loop
     */
    void start_event_loop();

    /**
     * @brief Stop the event processing loop
     */
    void stop_event_loop();

    /**
     * @brief Apply all settings (biases and features)
     */
    void apply_all_settings();

    /**
     * @brief Get access to the camera
     */
    Metavision::Camera* get_camera();

    /**
     * @brief Get bias manager
     */
    BiasManager& bias_manager() { return *bias_mgr_; }

    /**
     * @brief Get feature manager
     */
    FeatureManager& feature_manager() { return *feature_mgr_; }

    /**
     * @brief Initialize features with the connected camera
     */
    void initialize_features();

private:
    void setup_event_callbacks();
    void create_frame_generator();

    core::AppState& state_;
    AppConfig& config_;
    std::unique_ptr<BiasManager> bias_mgr_;
    std::unique_ptr<FeatureManager> feature_mgr_;
};

} // namespace EventCamera

#endif // CAMERA_CONTROLLER_H
