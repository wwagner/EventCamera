#ifndef GA_PANEL_H
#define GA_PANEL_H

#include "ui/ui_panel.h"

// Forward declarations
namespace core {
    class AppState;
}
class AppConfig;

namespace ui {

/**
 * @brief Genetic Algorithm optimization panel
 *
 * Provides UI for configuring and running the genetic algorithm
 * to optimize camera parameters.
 *
 * Note: This is a simplified panel. Full GA integration would require
 * additional controller classes as outlined in the refactoring plan.
 */
class GAPanel : public UIPanel {
public:
    GAPanel(core::AppState& state, AppConfig& config);

    void render() override;
    std::string title() const override { return "Genetic Algorithm"; }

private:
    void render_info();

    core::AppState& state_;
    AppConfig& config_;
};

} // namespace ui

#endif // GA_PANEL_H
