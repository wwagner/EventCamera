#include "core/display_settings.h"

namespace core {

void DisplaySettings::set_target_fps(int fps) {
    target_display_fps_.store(fps);
}

int DisplaySettings::get_target_fps() const {
    return target_display_fps_.load();
}

void DisplaySettings::set_image_size(int width, int height) {
    image_width_.store(width);
    image_height_.store(height);
}

int DisplaySettings::get_image_width() const {
    return image_width_.load();
}

int DisplaySettings::get_image_height() const {
    return image_height_.load();
}

void DisplaySettings::set_add_images_mode(bool enabled) {
    add_images_mode_.store(enabled);
}

bool DisplaySettings::get_add_images_mode() const {
    return add_images_mode_.load();
}

void DisplaySettings::set_flip_second_view(bool enabled) {
    flip_second_view_.store(enabled);
}

bool DisplaySettings::get_flip_second_view() const {
    return flip_second_view_.load();
}

void DisplaySettings::set_grayscale_mode(bool enabled) {
    grayscale_mode_.store(enabled);
}

bool DisplaySettings::get_grayscale_mode() const {
    return grayscale_mode_.load();
}

void DisplaySettings::set_binary_stream_mode(BinaryStreamMode mode) {
    binary_stream_mode_.store(static_cast<int>(mode));
}

DisplaySettings::BinaryStreamMode DisplaySettings::get_binary_stream_mode() const {
    return static_cast<BinaryStreamMode>(binary_stream_mode_.load());
}

void DisplaySettings::set_binary_stream_mode_2(BinaryStreamMode mode) {
    binary_stream_mode_2_.store(static_cast<int>(mode));
}

DisplaySettings::BinaryStreamMode DisplaySettings::get_binary_stream_mode_2() const {
    return static_cast<BinaryStreamMode>(binary_stream_mode_2_.load());
}

void DisplaySettings::set_display_mode(DisplayMode mode) {
    display_mode_.store(static_cast<int>(mode));
}

DisplaySettings::DisplayMode DisplaySettings::get_display_mode() const {
    return static_cast<DisplayMode>(display_mode_.load());
}

} // namespace core
