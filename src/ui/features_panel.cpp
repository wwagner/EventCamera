#include "ui/features_panel.h"
#include <imgui.h>

namespace ui {

FeaturesPanel::FeaturesPanel(EventCamera::FeatureManager& feature_mgr)
    : feature_mgr_(feature_mgr) {
}

void FeaturesPanel::render() {
    if (!visible_) return;

    ImGui::SetNextWindowPos(position_, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(size_, ImGuiCond_FirstUseEver);

    if (ImGui::Begin(title().c_str(), &visible_)) {
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Advanced Features");
        ImGui::Spacing();

        // FeatureManager handles all rendering
        feature_mgr_.render_all_ui();
    }
    ImGui::End();
}

} // namespace ui
