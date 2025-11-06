#include "optimization/ga_controller.h"
#include "core/app_state.h"
#include "app_config.h"
#include "camera/camera_controller.h"
#include <iostream>
#include <algorithm>

namespace optimization {

GAController::GAController(core::AppState& state,
                           AppConfig& config,
                           EventCamera::CameraController& camera_ctrl)
    : state_(state), config_(config), camera_ctrl_(camera_ctrl) {

    std::cout << "GAController: Initialized" << std::endl;
}

GAController::~GAController() {
    stop_optimization();
}

void GAController::start_optimization() {
    if (is_running_) {
        std::cout << "GAController: Optimization already running" << std::endl;
        return;
    }

    std::cout << "GAController: Starting genetic algorithm optimization..." << std::endl;

    // Initialize optimizer
    // Note: Full implementation would create and configure the optimizer here
    // For now, this is a placeholder structure

    is_running_ = true;
    current_generation_ = 0;

    // Start optimization thread
    optimizer_thread_ = std::make_unique<std::thread>(
        &GAController::optimization_thread, this);

    std::cout << "GAController: Optimization started" << std::endl;
}

void GAController::stop_optimization() {
    if (!is_running_) {
        return;
    }

    std::cout << "GAController: Stopping optimization..." << std::endl;
    is_running_ = false;

    // Wait for thread to finish
    if (optimizer_thread_ && optimizer_thread_->joinable()) {
        optimizer_thread_->join();
    }

    std::cout << "GAController: Optimization stopped" << std::endl;
}

void GAController::apply_best_parameters() {
    std::lock_guard<std::mutex> lock(genome_mutex_);

    std::cout << "GAController: Applying best parameters to camera..." << std::endl;

    // Clamp genome to hardware limits
    auto clamped_genome = clamp_genome_to_hardware(best_genome_);

    // Apply to camera settings
    auto& cam_settings = config_.camera_settings();
    cam_settings.bias_diff = clamped_genome.bias_diff;
    cam_settings.bias_refr = clamped_genome.bias_refr;
    cam_settings.bias_fo = clamped_genome.bias_fo;
    cam_settings.bias_hpf = clamped_genome.bias_hpf;
    cam_settings.accumulation_time_s = clamped_genome.accumulation_time_s;

    // Apply via camera controller
    camera_ctrl_.apply_all_settings();

    std::cout << "GAController: Best parameters applied (fitness: "
             << best_fitness_ << ")" << std::endl;
}

int GAController::get_total_generations() const {
    return config_.ga_settings().num_generations;
}

EventCameraGeneticOptimizer::Genome GAController::clamp_genome_to_hardware(
    const EventCameraGeneticOptimizer::Genome& genome) {

    EventCameraGeneticOptimizer::Genome clamped = genome;

    // Get hardware ranges from bias manager
    auto& bias_mgr = camera_ctrl_.bias_manager();
    const auto& bias_ranges = bias_mgr.get_bias_ranges();

    // Clamp biases to hardware limits
    if (bias_ranges.count("bias_diff")) {
        const auto& range = bias_ranges.at("bias_diff");
        clamped.bias_diff = std::clamp(genome.bias_diff, range.min, range.max);
    }

    if (bias_ranges.count("bias_refr")) {
        const auto& range = bias_ranges.at("bias_refr");
        clamped.bias_refr = std::clamp(genome.bias_refr, range.min, range.max);
    }

    if (bias_ranges.count("bias_fo")) {
        const auto& range = bias_ranges.at("bias_fo");
        clamped.bias_fo = std::clamp(genome.bias_fo, range.min, range.max);
    }

    if (bias_ranges.count("bias_hpf")) {
        const auto& range = bias_ranges.at("bias_hpf");
        clamped.bias_hpf = std::clamp(genome.bias_hpf, range.min, range.max);
    }

    // Clamp accumulation time to reasonable range
    clamped.accumulation_time_s = std::clamp(
        genome.accumulation_time_s, 0.001f, 0.1f);

    return clamped;
}

void GAController::optimization_thread() {
    std::cout << "GAController: Optimization thread started" << std::endl;

    // Note: Full implementation would run the genetic algorithm here
    // For now, this is a placeholder that demonstrates the structure

    int total_generations = get_total_generations();

    for (int gen = 0; gen < total_generations && is_running_; ++gen) {
        current_generation_ = gen;

        // Simulate optimization work
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Update progress
        if (gen % 5 == 0) {
            std::cout << "GAController: Generation " << gen << "/"
                     << total_generations << std::endl;
        }
    }

    current_generation_ = total_generations;
    is_running_ = false;

    std::cout << "GAController: Optimization complete" << std::endl;
}

} // namespace optimization
