#pragma once

#include <metavision/sdk/driver/camera.h>
#include <string>
#include <vector>
#include <memory>

/**
 * CameraManager handles enumeration, selection, and initialization of SilkyEvCam event cameras.
 * Supports both single and dual camera configurations with optional serial number selection.
 *
 * Hardware: CenturyArks SilkyEvCam HD
 * SDK: Metavision (via CenturyArks silky_common_plugin)
 */
class CameraManager {
public:
    struct CameraInfo {
        std::string serial;
        uint16_t width;
        uint16_t height;
        std::unique_ptr<Metavision::Camera> camera;

        CameraInfo(const std::string& s, uint16_t w, uint16_t h, std::unique_ptr<Metavision::Camera> cam)
            : serial(s), width(w), height(h), camera(std::move(cam)) {}
    };

    /**
     * Initialize cameras based on serial numbers.
     * @param serial1 Serial number for first camera (empty = auto-detect)
     * @param serial2 Serial number for second camera (empty = no second camera)
     * @return Number of cameras successfully initialized
     */
    int initialize(const std::string& serial1 = "", const std::string& serial2 = "");

    /**
     * Get number of initialized cameras
     */
    int num_cameras() const { return static_cast<int>(cameras_.size()); }

    /**
     * Get camera info by index
     */
    CameraInfo& get_camera(int index) { return cameras_[index]; }
    const CameraInfo& get_camera(int index) const { return cameras_[index]; }

    /**
     * Get all cameras
     */
    std::vector<CameraInfo>& get_cameras() { return cameras_; }
    const std::vector<CameraInfo>& get_cameras() const { return cameras_; }

    /**
     * List all available camera serial numbers
     */
    static std::vector<std::string> list_available_cameras();

private:
    std::vector<CameraInfo> cameras_;

    /**
     * Open camera by serial number or index
     */
    std::unique_ptr<Metavision::Camera> open_camera(const std::string& serial);

    /**
     * Open first available camera
     */
    std::unique_ptr<Metavision::Camera> open_first_available();
};
