#ifndef FEATURES_PANEL_H
#define FEATURES_PANEL_H

#include "ui/ui_panel.h"
#include "camera/feature_manager.h"

namespace ui {

/**
 * @brief Panel for hardware features
 *
 * Displays all available camera hardware features using the FeatureManager.
 * Features are automatically organized by category.
 */
class FeaturesPanel : public UIPanel {
public:
    FeaturesPanel(EventCamera::FeatureManager& feature_mgr);

    void render() override;
    std::string title() const override { return "Advanced Features"; }

private:
    EventCamera::FeatureManager& feature_mgr_;
};

} // namespace ui

#endif // FEATURES_PANEL_H
