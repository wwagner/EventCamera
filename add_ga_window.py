#!/usr/bin/env python3
"""Add GA optimization window to main.cpp"""

with open('src/main.cpp', 'r') as f:
    content = f.read()

ga_window = '''
        // Genetic Algorithm Optimization window
        ImGui::SetNextWindowPos(ImVec2(10, 870), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(420, 400), ImGuiCond_FirstUseEver);

        if (ImGui::Begin("Genetic Algorithm Optimization")) {
            ImGui::TextWrapped("Optimize camera parameters using genetic algorithm");
            ImGui::Separator();

            auto& ga_cfg = config.ga_settings();

            if (!ga_state.running) {
                ImGui::Text("Configuration (from event_config.ini)");
                ImGui::Text("Population: %d | Generations: %d", ga_cfg.population_size, ga_cfg.num_generations);
                ImGui::Text("Mutation: %.2f | Crossover: %.2f", ga_cfg.mutation_rate, ga_cfg.crossover_rate);
                ImGui::Text("Frames/Eval: %d", ga_cfg.frames_per_eval);

                ImGui::Separator();
                ImGui::Text("Parameters to Optimize:");
                ImGui::Text("(Edit event_config.ini to change)");

                ImGui::BeginDisabled();
                bool opt_bd = ga_cfg.optimize_bias_diff;
                bool opt_br = ga_cfg.optimize_bias_refr;
                bool opt_bf = ga_cfg.optimize_bias_fo;
                bool opt_bh = ga_cfg.optimize_bias_hpf;
                bool opt_bp = ga_cfg.optimize_bias_pr;
                bool opt_ac = ga_cfg.optimize_accumulation;
                bool opt_tf = ga_cfg.optimize_trail_filter;
                bool opt_af = ga_cfg.optimize_antiflicker;
                bool opt_er = ga_cfg.optimize_erc;

                ImGui::Checkbox("bias_diff", &opt_bd); ImGui::SameLine();
                ImGui::Checkbox("bias_refr", &opt_br);
                ImGui::Checkbox("bias_fo", &opt_bf); ImGui::SameLine();
                ImGui::Checkbox("bias_hpf", &opt_bh);
                ImGui::Checkbox("bias_pr", &opt_bp); ImGui::SameLine();
                ImGui::Checkbox("accumulation", &opt_ac);
                ImGui::Checkbox("trail_filter", &opt_tf); ImGui::SameLine();
                ImGui::Checkbox("antiflicker", &opt_af);
                ImGui::Checkbox("erc", &opt_er);
                ImGui::EndDisabled();

                ImGui::Separator();

                if (camera_state.camera_connected) {
                    if (ImGui::Button("Start Optimization", ImVec2(-1, 0))) {
                        std::cout << "\\n=== Starting GA Optimization ===" << std::endl;

                        // TODO: Implement GA optimization
                        // This will require proper integration with EventCameraGeneticOptimizer
                        ImGui::OpenPopup("GA Not Implemented");
                    }
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Connect camera to start optimization");
                }

                if (ImGui::BeginPopupModal("GA Not Implemented", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
                    ImGui::Text("Genetic algorithm optimization is not yet fully implemented.");
                    ImGui::Text("The GA engine exists but needs proper integration with");
                    ImGui::Text("the fitness callback and parameter application.");
                    ImGui::Separator();
                    if (ImGui::Button("OK", ImVec2(120, 0))) {
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::EndPopup();
                }
            } else {
                // Show progress (placeholder for when GA is running)
                ImGui::Text("Optimization Running...");
                ImGui::ProgressBar(0.5f, ImVec2(-1, 0));
                ImGui::Text("Generation: %d / %d", ga_state.current_generation.load(), ga_cfg.num_generations);
                ImGui::Text("Best Fitness: %.4f", ga_state.best_fitness.load());

                if (ImGui::Button("Stop Optimization", ImVec2(-1, 0))) {
                    ga_state.running = false;
                }
            }

            ImGui::Separator();

            // Display best results
            if (ga_state.best_fitness < 1e9f) {
                ImGui::Text("Best Results");
                ImGui::Text("Fitness: %.4f", ga_state.best_fitness.load());
                ImGui::Text("Contrast: %.2f", ga_state.best_result.contrast_score);
                ImGui::Text("Noise: %.4f", ga_state.best_result.noise_metric);

                if (ImGui::CollapsingHeader("Best Parameters")) {
                    ImGui::Text("Biases:");
                    ImGui::Text("  diff=%d refr=%d fo=%d",
                               ga_state.best_genome.bias_diff,
                               ga_state.best_genome.bias_refr,
                               ga_state.best_genome.bias_fo);
                    ImGui::Text("  hpf=%d pr=%d",
                               ga_state.best_genome.bias_hpf,
                               ga_state.best_genome.bias_pr);
                    ImGui::Text("Accumulation: %.3f s", ga_state.best_genome.accumulation_time_s);
                }

                if (!ga_state.running && camera_state.camera_connected) {
                    if (ImGui::Button("Apply Best Parameters", ImVec2(-1, 0))) {
                        // Apply to config and camera
                        config.camera_settings().bias_diff = ga_state.best_genome.bias_diff;
                        config.camera_settings().bias_refr = ga_state.best_genome.bias_refr;
                        config.camera_settings().bias_fo = ga_state.best_genome.bias_fo;
                        config.camera_settings().bias_hpf = ga_state.best_genome.bias_hpf;
                        config.camera_settings().bias_pr = ga_state.best_genome.bias_pr;
                        config.camera_settings().accumulation_time_s = ga_state.best_genome.accumulation_time_s;

                        auto& cam_info = camera_state.camera_mgr->get_camera(0);
                        apply_bias_settings(*cam_info.camera, config.camera_settings());

                        std::cout << "Applied best GA parameters to camera" << std::endl;
                    }
                }
            } else {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No optimization results yet");
            }
        }
        ImGui::End();
'''

# Find the insertion point after Camera Feed window
marker = '        ImGui::End();\n\n        // Render'
if marker in content and 'Genetic Algorithm Optimization' not in content:
    content = content.replace(marker, '        ImGui::End();\n' + ga_window + '\n        // Render')

    with open('src/main.cpp', 'w') as f:
        f.write(content)

    print("GA window added successfully")
elif 'Genetic Algorithm Optimization' in content:
    print("GA window already exists")
else:
    print("Could not find insertion point")
