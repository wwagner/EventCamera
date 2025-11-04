#include "camera_manager.h"
#include <metavision/hal/device/device_discovery.h>
#include <iostream>
#include <stdexcept>

int CameraManager::initialize(const std::string& serial1, const std::string& serial2) {
    cameras_.clear();

    try {
        // Case 1: Both serials specified
        if (!serial1.empty() && !serial2.empty()) {
            std::cout << "Opening cameras with serial numbers: " << serial1 << ", " << serial2 << std::endl;

            auto cam1 = open_camera(serial1);
            if (cam1) {
                const auto& geom1 = cam1->geometry();
                cameras_.emplace_back(serial1, geom1.width(), geom1.height(), std::move(cam1));
                std::cout << "Camera 1 opened: " << serial1 << " (" << geom1.width() << "x" << geom1.height() << ")" << std::endl;
            }

            auto cam2 = open_camera(serial2);
            if (cam2) {
                const auto& geom2 = cam2->geometry();
                cameras_.emplace_back(serial2, geom2.width(), geom2.height(), std::move(cam2));
                std::cout << "Camera 2 opened: " << serial2 << " (" << geom2.width() << "x" << geom2.height() << ")" << std::endl;
            }
        }
        // Case 2: Only first serial specified
        else if (!serial1.empty() && serial2.empty()) {
            std::cout << "Opening camera with serial number: " << serial1 << std::endl;

            auto cam1 = open_camera(serial1);
            if (cam1) {
                const auto& geom1 = cam1->geometry();
                cameras_.emplace_back(serial1, geom1.width(), geom1.height(), std::move(cam1));
                std::cout << "Camera opened: " << serial1 << " (" << geom1.width() << "x" << geom1.height() << ")" << std::endl;
            }
        }
        // Case 3: Auto-detect (no serials specified)
        else {
            std::cout << "Auto-detecting cameras..." << std::endl;

            auto available = list_available_cameras();
            std::cout << "Found " << available.size() << " camera(s)" << std::endl;

            if (available.empty()) {
                std::cerr << "No cameras detected!" << std::endl;
                return 0;
            }

            // Open first camera
            auto cam1 = open_camera(available[0]);
            if (cam1) {
                const auto& geom1 = cam1->geometry();
                cameras_.emplace_back(available[0], geom1.width(), geom1.height(), std::move(cam1));
                std::cout << "Camera 1 opened: " << available[0] << " (" << geom1.width() << "x" << geom1.height() << ")" << std::endl;
            }

            // Open second camera if available
            if (available.size() >= 2) {
                auto cam2 = open_camera(available[1]);
                if (cam2) {
                    const auto& geom2 = cam2->geometry();
                    cameras_.emplace_back(available[1], geom2.width(), geom2.height(), std::move(cam2));
                    std::cout << "Camera 2 opened: " << available[1] << " (" << geom2.width() << "x" << geom2.height() << ")" << std::endl;
                }
            }
        }

    } catch (const Metavision::CameraException& e) {
        std::cerr << "Camera initialization error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return num_cameras();
}

std::vector<std::string> CameraManager::list_available_cameras() {
    std::vector<std::string> serials;

    try {
        // Use list() to get camera serials
        auto serial_list = Metavision::DeviceDiscovery::list();

        for (const auto& serial : serial_list) {
            serials.push_back(serial);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error listing cameras: " << e.what() << std::endl;
    }

    return serials;
}

std::unique_ptr<Metavision::Camera> CameraManager::open_camera(const std::string& serial) {
    try {
        auto camera = std::make_unique<Metavision::Camera>(
            Metavision::Camera::from_serial(serial)
        );
        return camera;
    } catch (const Metavision::CameraException& e) {
        std::cerr << "Failed to open camera " << serial << ": " << e.what() << std::endl;
        return nullptr;
    }
}

std::unique_ptr<Metavision::Camera> CameraManager::open_first_available() {
    try {
        auto camera = std::make_unique<Metavision::Camera>(
            Metavision::Camera::from_first_available()
        );
        return camera;
    } catch (const Metavision::CameraException& e) {
        std::cerr << "Failed to open first available camera: " << e.what() << std::endl;
        return nullptr;
    }
}
