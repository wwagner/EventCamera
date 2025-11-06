#ifndef GA_CONTROLLER_H
#define GA_CONTROLLER_H

#include "event_camera_genetic_optimizer.h"
#include <memory>
#include <thread>
#include <atomic>

// Forward declarations
namespace core {
    class AppState;
}
class AppConfig;

namespace EventCamera {
    class CameraController;
}

namespace optimization {

/**
 * @brief Genetic Algorithm controller
 *
 * Manages the lifecycle and integration of genetic algorithm optimization
 * for camera parameters. Handles genome clamping, optimization execution,
 * and result application.
 */
class GAController {
public:
    GAController(core::AppState& state,
                 AppConfig& config,
                 EventCamera::CameraController& camera_ctrl);
    ~GAController();

    /**
     * @brief Start optimization process
     */
    void start_optimization();

    /**
     * @brief Stop optimization process
     */
    void stop_optimization();

    /**
     * @brief Check if optimization is running
     */
    bool is_running() const { return is_running_; }

    /**
     * @brief Get best fitness achieved
     */
    float get_best_fitness() const { return best_fitness_; }

    /**
     * @brief Get best genome found
     */
    const EventCameraGeneticOptimizer::Genome& get_best_genome() const {
        return best_genome_;
    }

    /**
     * @brief Apply best parameters to camera
     */
    void apply_best_parameters();

    /**
     * @brief Get current generation number
     */
    int get_current_generation() const { return current_generation_; }

    /**
     * @brief Get total generations configured
     */
    int get_total_generations() const;

private:
    /**
     * @brief Clamp genome values to hardware limits
     */
    EventCameraGeneticOptimizer::Genome clamp_genome_to_hardware(
        const EventCameraGeneticOptimizer::Genome& genome);

    /**
     * @brief Optimization thread function
     */
    void optimization_thread();

    core::AppState& state_;
    AppConfig& config_;
    EventCamera::CameraController& camera_ctrl_;

    std::unique_ptr<EventCameraGeneticOptimizer> optimizer_;
    std::unique_ptr<std::thread> optimizer_thread_;

    std::atomic<bool> is_running_{false};
    std::atomic<int> current_generation_{0};
    std::atomic<float> best_fitness_{0.0f};

    EventCameraGeneticOptimizer::Genome best_genome_;
    std::mutex genome_mutex_;
};

} // namespace optimization

#endif // GA_CONTROLLER_H
