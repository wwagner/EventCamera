#pragma once

#include <vector>
#include <functional>
#include <chrono>
#include <string>
#include "event_camera_genetic_optimizer.h"

/**
 * SensitivityAnalyzer - Analyzes parameter sensitivity for genetic optimization
 *
 * Computes numerical gradients (partial derivatives) to measure how changes in
 * each parameter affect the fitness function. This information can be used to:
 * - Understand which parameters have the most impact
 * - Scale mutation rates inversely to sensitivity
 * - Guide adaptive optimization strategies
 *
 * Phase 1: Basic sensitivity analysis with logging (no mutation scaling yet)
 */
class SensitivityAnalyzer {
public:
    /**
     * Parameter indices for the reduced parameter set
     */
    enum ParameterIndex {
        BIAS_DIFF_ON = 0,
        BIAS_DIFF_OFF = 1,
        BIAS_HPF = 2,
        TRAIL_THRESHOLD = 3,
        NUM_PARAMS = 4
    };

    /**
     * Result of sensitivity analysis for one parameter
     */
    struct ParameterSensitivity {
        ParameterIndex param_index;
        std::string param_name;
        double sensitivity;           // |df/dx| - absolute gradient magnitude
        double normalization_factor;  // Inverse scaling factor (high sensitivity = low factor)
        double base_value;            // Parameter value at analysis point
        double delta_used;            // Delta used for gradient computation
        double fitness_plus;          // Fitness at base + delta
        double fitness_minus;         // Fitness at base - delta
    };

    /**
     * Complete sensitivity analysis result
     */
    struct SensitivityResult {
        std::vector<ParameterSensitivity> param_sensitivities;
        double base_fitness;                    // Fitness at base genome
        double max_sensitivity;                 // Maximum sensitivity across all params
        double min_sensitivity;                 // Minimum sensitivity across all params
        std::chrono::system_clock::time_point timestamp;
        int generation;                         // Generation when analysis was performed

        SensitivityResult()
            : base_fitness(0.0)
            , max_sensitivity(0.0)
            , min_sensitivity(1e9)
            , timestamp(std::chrono::system_clock::now())
            , generation(0)
        {}
    };

    /**
     * Configuration for sensitivity analysis
     */
    struct Config {
        double delta_ratio = 0.01;              // Delta as fraction of parameter range (1%)
        bool use_central_difference = true;     // Use central difference (more accurate)
        bool verbose_logging = true;            // Print detailed output
        std::string log_file = "sensitivity_analysis_log.csv";

        // Phase 2: Normalization settings
        double min_normalization_factor = 0.1;  // Minimum scaling (prevents too small mutations)
        double max_normalization_factor = 10.0; // Maximum scaling (prevents too large mutations)
    };

    /**
     * Constructor
     */
    SensitivityAnalyzer(const Config& config = Config());

    /**
     * Perform sensitivity analysis on a genome
     *
     * @param genome Base genome to analyze
     * @param fitness_evaluator Function that evaluates genome fitness
     * @param generation Current generation number (for logging)
     * @return Sensitivity analysis result
     */
    SensitivityResult analyze(
        const EventCameraGeneticOptimizer::Genome& genome,
        std::function<float(const EventCameraGeneticOptimizer::Genome&)> fitness_evaluator,
        int generation = 0
    );

    /**
     * Log sensitivity result to CSV file
     * Creates header on first call
     */
    void log_result(const SensitivityResult& result);

    /**
     * Print sensitivity result to console
     */
    void print_result(const SensitivityResult& result) const;

private:
    Config config_;
    bool log_header_written_;

    /**
     * Compute numerical gradient for a single parameter
     *
     * @param base_genome Genome to perturb
     * @param param_index Which parameter to perturb
     * @param delta Perturbation magnitude
     * @param fitness_evaluator Function to evaluate fitness
     * @return Gradient (sensitivity) value
     */
    double compute_gradient(
        const EventCameraGeneticOptimizer::Genome& base_genome,
        ParameterIndex param_index,
        double delta,
        std::function<float(const EventCameraGeneticOptimizer::Genome&)> fitness_evaluator,
        double& fitness_plus,  // Output: fitness at +delta
        double& fitness_minus  // Output: fitness at -delta
    );

    /**
     * Apply parameter perturbation to genome
     */
    void perturb_parameter(
        EventCameraGeneticOptimizer::Genome& genome,
        ParameterIndex param_index,
        double delta
    ) const;

    /**
     * Get parameter name as string
     */
    std::string get_parameter_name(ParameterIndex param_index) const;

    /**
     * Get parameter value from genome
     */
    double get_parameter_value(
        const EventCameraGeneticOptimizer::Genome& genome,
        ParameterIndex param_index
    ) const;

    /**
     * Get parameter range (max - min)
     */
    double get_parameter_range(
        const EventCameraGeneticOptimizer::Genome& genome,
        ParameterIndex param_index
    ) const;

    /**
     * Compute normalization factors based on sensitivities
     * Inverse relationship: high sensitivity = low factor (smaller mutations)
     */
    void compute_normalization_factors(SensitivityResult& result);
};
