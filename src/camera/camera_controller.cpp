#include "camera/camera_controller.h"
#include "core/app_state.h"
#include "app_config.h"
#include "camera_manager.h"
#include <iostream>

namespace EventCamera {

CameraController::CameraController(core::AppState& state, AppConfig& config)
    : state_(state), config_(config) {

    // Create managers
    bias_mgr_ = std::make_unique<BiasManager>();
    feature_mgr_ = std::make_unique<FeatureManager>();
}

CameraController::~CameraController() {
    stop_event_loop();
    disconnect_camera();
}

bool CameraController::connect_camera(const std::string& serial_hint) {
    std::cout << "CameraController: Attempting to connect to camera..." << std::endl;

    if (!state_.camera_state().camera_manager()) {
        std::cerr << "CameraController: No camera manager available" << std::endl;
        return false;
    }

    try {
        // Camera connection is handled by CameraManager
        // This is a simplified version - full integration would require more work
        if (state_.camera_state().is_connected()) {
            std::cout << "CameraController: Camera already connected" << std::endl;
            return true;
        }

        std::cout << "CameraController: Camera connection would be initiated here" << std::endl;
        return false;

    } catch (const std::exception& e) {
        std::cerr << "CameraController: Connection failed: " << e.what() << std::endl;
        return false;
    }
}

void CameraController::disconnect_camera() {
    std::cout << "CameraController: Disconnecting camera..." << std::endl;

    // Stop event loop first
    stop_event_loop();

    // Shutdown features
    if (feature_mgr_) {
        feature_mgr_->shutdown_all();
    }

    // Camera disconnection is handled by CameraManager lifecycle
    std::cout << "CameraController: Camera disconnected" << std::endl;
}

bool CameraController::is_connected() const {
    return state_.camera_state().is_connected();
}

void CameraController::start_event_loop() {
    std::cout << "CameraController: Event loop management would be handled here" << std::endl;
    // Note: In full implementation, this would start the event processing thread
}

void CameraController::stop_event_loop() {
    std::cout << "CameraController: Stopping event loop..." << std::endl;
    // Note: In full implementation, this would stop and join the event thread
}

void CameraController::apply_all_settings() {
    std::cout << "CameraController: Applying all settings..." << std::endl;

    // Apply biases
    if (bias_mgr_ && bias_mgr_->is_initialized()) {
        bias_mgr_->apply_to_camera();
    }

    // Apply feature settings
    if (feature_mgr_) {
        feature_mgr_->apply_all_settings();
    }
}

Metavision::Camera* CameraController::get_camera() {
    if (!state_.camera_state().is_connected() || !state_.camera_state().camera_manager()) {
        return nullptr;
    }

    auto& cam_info = state_.camera_state().camera_manager()->get_camera(0);
    return cam_info.camera.get();
}

void CameraController::initialize_features() {
    if (!is_connected()) {
        std::cerr << "CameraController: Cannot initialize features - camera not connected" << std::endl;
        return;
    }

    auto* camera = get_camera();
    if (!camera) {
        std::cerr << "CameraController: Cannot get camera instance" << std::endl;
        return;
    }

    // Initialize bias manager
    std::cout << "CameraController: Initializing bias manager..." << std::endl;
    bias_mgr_->initialize(*camera);

    // Initialize feature manager
    std::cout << "CameraController: Initializing features..." << std::endl;
    feature_mgr_->initialize_all(*camera);
}

void CameraController::setup_event_callbacks() {
    // Note: Event callback setup would be implemented here in full version
    std::cout << "CameraController: Event callbacks would be set up here" << std::endl;
}

void CameraController::create_frame_generator() {
    // Note: Frame generator creation would be implemented here in full version
    std::cout << "CameraController: Frame generator would be created here" << std::endl;
}

} // namespace EventCamera
