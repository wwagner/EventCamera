#include "ui/ga_panel.h"
#include "core/app_state.h"
#include "app_config.h"
#include <imgui.h>

namespace ui {

GAPanel::GAPanel(core::AppState& state, AppConfig& config)
    : state_(state), config_(config) {

    // Set default position to avoid overlapping other panels
    position_ = ImVec2(10, 700);
    size_ = ImVec2(400, 300);
}

void GAPanel::render() {
    if (!visible_) return;

    ImGui::SetNextWindowPos(position_, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(size_, ImGuiCond_FirstUseEver);

    if (ImGui::Begin(title().c_str(), &visible_)) {
        render_info();
    }
    ImGui::End();
}

void GAPanel::render_info() {
    ImGui::TextWrapped("Genetic Algorithm Optimization");
    ImGui::Spacing();

    ImGui::TextWrapped("The GA optimizer can automatically tune camera parameters "
                      "for optimal performance.");
    ImGui::Spacing();

    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                      "Note: Full GA integration will be completed in Phase 6");
    ImGui::Spacing();

    // Show current GA configuration
    auto& ga_config = config_.ga_settings();
    ImGui::Text("Configuration:");
    ImGui::BulletText("Population: %d", ga_config.population_size);
    ImGui::BulletText("Generations: %d", ga_config.num_generations);
    ImGui::BulletText("Mutation rate: %.3f", ga_config.mutation_rate);

    ImGui::Spacing();
    ImGui::TextWrapped("Parameters to optimize:");
    ImGui::BulletText("Biases: %s", ga_config.optimize_bias_diff ? "Yes" : "No");
    ImGui::BulletText("Trail filter: %s", ga_config.optimize_trail_filter ? "Yes" : "No");
    ImGui::BulletText("Anti-flicker: %s", ga_config.optimize_antiflicker ? "Yes" : "No");

    ImGui::Spacing();
    ImGui::TextWrapped("Edit event_config.ini to change GA settings.");
}

} // namespace ui
