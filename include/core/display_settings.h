#pragma once

#include <atomic>
#include <cstdint>

namespace core {

/**
 * Display configuration settings
 *
 * Manages display rate and window size settings.
 */
class DisplaySettings {
public:
    DisplaySettings() = default;
    ~DisplaySettings() = default;

    // Non-copyable
    DisplaySettings(const DisplaySettings&) = delete;
    DisplaySettings& operator=(const DisplaySettings&) = delete;

    /**
     * Set target display FPS
     * @param fps Target frames per second
     */
    void set_target_fps(int fps);

    /**
     * Get target display FPS
     * @return Frames per second
     */
    int get_target_fps() const;

    /**
     * Set image dimensions
     * @param width Image width in pixels
     * @param height Image height in pixels
     */
    void set_image_size(int width, int height);

    /**
     * Get image width
     * @return Width in pixels
     */
    int get_image_width() const;

    /**
     * Get image height
     * @return Height in pixels
     */
    int get_image_height() const;

private:
    std::atomic<int> target_display_fps_{10};
    std::atomic<int> image_width_{1280};
    std::atomic<int> image_height_{720};
};

} // namespace core
