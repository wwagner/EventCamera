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

    /**
     * Set add images mode (combine both cameras into one view)
     * @param enabled true to combine cameras, false for separate views
     */
    void set_add_images_mode(bool enabled);

    /**
     * Get add images mode status
     * @return true if cameras are combined, false if separate
     */
    bool get_add_images_mode() const;

    /**
     * Set flip second view mode (flip camera 1 horizontally)
     * @param enabled true to flip horizontally, false for normal
     */
    void set_flip_second_view(bool enabled);

    /**
     * Get flip second view status
     * @return true if camera 1 is flipped horizontally, false if normal
     */
    bool get_flip_second_view() const;

    /**
     * Set grayscale mode (convert BGR to single-channel grayscale)
     * @param enabled true for grayscale, false for BGR color
     */
    void set_grayscale_mode(bool enabled);

    /**
     * Get grayscale mode status
     * @return true if outputting grayscale, false if BGR color
     */
    bool get_grayscale_mode() const;

    /**
     * Binary stream modes for early 1-bit conversion
     */
    enum class BinaryStreamMode {
        OFF = 0,        // No binary conversion (8-bit passthrough)
        DOWN = 1,       // Range 3 [96-127] only
        UP = 2,         // Range 7 [224-255] only
        UP_DOWN = 3     // Both ranges combined
    };

    /**
     * Set binary stream mode
     * @param mode Stream mode (OFF, DOWN, UP, UP_DOWN)
     */
    void set_binary_stream_mode(BinaryStreamMode mode);

    /**
     * Get binary stream mode
     * @return Current stream mode
     */
    BinaryStreamMode get_binary_stream_mode() const;

private:
    std::atomic<int> target_display_fps_{10};
    std::atomic<int> image_width_{1280};
    std::atomic<int> image_height_{720};
    std::atomic<bool> add_images_mode_{false};
    std::atomic<bool> flip_second_view_{false};
    std::atomic<bool> grayscale_mode_{false};
    std::atomic<int> binary_stream_mode_{0};  // BinaryStreamMode as int (default: OFF)
};

} // namespace core
