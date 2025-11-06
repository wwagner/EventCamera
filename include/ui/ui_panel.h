#ifndef UI_PANEL_H
#define UI_PANEL_H

#include <string>
#include <imgui.h>

namespace ui {

/**
 * @brief Base class for all UI panels
 *
 * Provides common functionality for ImGui panels including
 * visibility, position, and size management.
 */
class UIPanel {
public:
    virtual ~UIPanel() = default;

    /**
     * @brief Render the panel (called each frame)
     *
     * This method should contain all ImGui rendering code for the panel.
     */
    virtual void render() = 0;

    /**
     * @brief Get the panel title
     * @return Title string for the panel window
     */
    virtual std::string title() const = 0;

    /**
     * @brief Set panel visibility
     * @param visible true to show, false to hide
     */
    void set_visible(bool visible) { visible_ = visible; }

    /**
     * @brief Check if panel is visible
     * @return true if visible
     */
    bool is_visible() const { return visible_; }

    /**
     * @brief Set panel position
     * @param pos Position as ImVec2
     */
    void set_position(ImVec2 pos) { position_ = pos; }

    /**
     * @brief Get panel position
     * @return Position as ImVec2
     */
    ImVec2 get_position() const { return position_; }

    /**
     * @brief Set panel size
     * @param size Size as ImVec2
     */
    void set_size(ImVec2 size) { size_ = size; }

    /**
     * @brief Get panel size
     * @return Size as ImVec2
     */
    ImVec2 get_size() const { return size_; }

protected:
    bool visible_ = true;
    ImVec2 position_{10, 10};
    ImVec2 size_{400, 600};
};

} // namespace ui

#endif // UI_PANEL_H
