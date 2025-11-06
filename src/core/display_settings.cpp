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

} // namespace core
