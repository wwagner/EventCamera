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
     * Bit selection mode for bit extraction (0-7 only)
     */
    enum class BinaryStreamMode {
        BIT_0 = 0,      // Extract bit 0 only
        BIT_1 = 1,      // Extract bit 1 only
        BIT_2 = 2,      // Extract bit 2 only
        BIT_3 = 3,      // Extract bit 3 only
        BIT_4 = 4,      // Extract bit 4 only
        BIT_5 = 5,      // Extract bit 5 only
        BIT_6 = 6,      // Extract bit 6 only
        BIT_7 = 7       // Extract bit 7 only
    };

    /**
     * Display mode for dual bit processing
     */
    enum class DisplayMode {
        OR_BEFORE_PROCESSING = 0,  // OR bits before processing, display combined
        OR_AFTER_PROCESSING = 1,   // Process separately, OR after, display combined
        DISPLAY_BIT_1 = 2,         // Display only first bit selector result
        DISPLAY_BIT_2 = 3          // Display only second bit selector result
    };

    /**
     * Set binary stream mode (first bit selector)
     * @param mode Stream mode (ALL_BITS or BIT_0 through BIT_7)
     */
    void set_binary_stream_mode(BinaryStreamMode mode);

    /**
     * Get binary stream mode (first bit selector)
     * @return Current stream mode
     */
    BinaryStreamMode get_binary_stream_mode() const;

    /**
     * Set second binary stream mode (second bit selector)
     * @param mode Stream mode (ALL_BITS or BIT_0 through BIT_7)
     */
    void set_binary_stream_mode_2(BinaryStreamMode mode);

    /**
     * Get second binary stream mode (second bit selector)
     * @return Current stream mode for second selector
     */
    BinaryStreamMode get_binary_stream_mode_2() const;

    /**
     * Set display mode (how to combine/display dual bits)
     * @param mode Display mode selection
     */
    void set_display_mode(DisplayMode mode);

    /**
     * Get display mode
     * @return Current display mode
     */
    DisplayMode get_display_mode() const;

    /**
     * Set red pixel percentage (for combined view statistics)
     * @param percentage Percentage of non-black pixels that are red (0.0 to 100.0)
     */
    void set_red_pixel_percentage(float percentage);

    /**
     * Get red pixel percentage
     * @return Percentage of non-black pixels that are red (0.0 to 100.0)
     */
    float get_red_pixel_percentage() const;

private:
    std::atomic<int> target_display_fps_{10};
    std::atomic<int> image_width_{1280};
    std::atomic<int> image_height_{720};
    std::atomic<bool> add_images_mode_{false};
    std::atomic<bool> flip_second_view_{false};
    std::atomic<bool> grayscale_mode_{false};
    std::atomic<int> binary_stream_mode_{0};  // BinaryStreamMode as int (default: BIT_0)
    std::atomic<int> binary_stream_mode_2_{7};  // Second bit selector (default: BIT_7)
    std::atomic<int> display_mode_{0};  // DisplayMode as int (default: OR_BEFORE_PROCESSING)
    std::atomic<float> red_pixel_percentage_{0.0f};  // Percentage of non-black pixels that are red
};

} // namespace core
