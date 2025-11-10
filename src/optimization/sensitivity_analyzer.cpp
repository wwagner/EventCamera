#include "optimization/sensitivity_analyzer.h"
#include "event_camera_genetic_optimizer.h"  // Full definition needed for Genome access
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace std;

// ============================================================================
// Constructor
// ============================================================================

SensitivityAnalyzer::SensitivityAnalyzer(const Config& config)
    : config_(config)
    , log_header_written_(false)
{
}

// ============================================================================
// Main Analysis Function
// ============================================================================

SensitivityAnalyzer::SensitivityResult SensitivityAnalyzer::analyze(
    const EventCameraGeneticOptimizer::Genome& genome,
    function<float(const EventCameraGeneticOptimizer::Genome&)> fitness_evaluator,
    int generation)
{
    if (config_.verbose_logging) {
        cout << "\n=======================================" << endl;
        cout << "Performing Sensitivity Analysis" << endl;
        cout << "Generation: " << generation << endl;
        cout << "=======================================" << endl;
    }

    SensitivityResult result;
    result.generation = generation;
    result.timestamp = chrono::system_clock::now();

    // Evaluate base fitness
    result.base_fitness = fitness_evaluator(genome);

    if (config_.verbose_logging) {
        cout << "Base fitness: " << result.base_fitness << endl;
        cout << "\nAnalyzing parameters:" << endl;
    }

    // Analyze each parameter
    for (int i = 0; i < NUM_PARAMS; ++i) {
        ParameterIndex param_idx = static_cast<ParameterIndex>(i);

        ParameterSensitivity param_sens;
        param_sens.param_index = param_idx;
        param_sens.param_name = get_parameter_name(param_idx);
        param_sens.base_value = get_parameter_value(genome, param_idx);

        // Compute delta as fraction of parameter range
        double range = get_parameter_range(genome, param_idx);
        param_sens.delta_used = config_.delta_ratio * range;

        // Compute gradient
        param_sens.sensitivity = compute_gradient(
            genome,
            param_idx,
            param_sens.delta_used,
            fitness_evaluator,
            param_sens.fitness_plus,
            param_sens.fitness_minus
        );

        result.param_sensitivities.push_back(param_sens);

        if (config_.verbose_logging) {
            cout << "  " << param_sens.param_name << ": "
                 << "sensitivity=" << fixed << setprecision(6) << param_sens.sensitivity
                 << " (base=" << param_sens.base_value
                 << ", range=" << range
                 << ", delta=" << param_sens.delta_used << ")" << endl;
        }
    }

    // Compute statistics
    result.max_sensitivity = 0.0;
    result.min_sensitivity = 1e9;

    for (const auto& ps : result.param_sensitivities) {
        result.max_sensitivity = max(result.max_sensitivity, ps.sensitivity);
        result.min_sensitivity = min(result.min_sensitivity, ps.sensitivity);
    }

    // Compute normalization factors (Phase 2)
    compute_normalization_factors(result);

    if (config_.verbose_logging) {
        cout << "\nSensitivity Range:" << endl;
        cout << "  Max: " << result.max_sensitivity << endl;
        cout << "  Min: " << result.min_sensitivity << endl;
        cout << "  Ratio: " << (result.max_sensitivity / max(result.min_sensitivity, 1e-10)) << "x" << endl;

        cout << "\nNormalization Factors (for adaptive mutation):" << endl;
        for (const auto& ps : result.param_sensitivities) {
            cout << "  " << ps.param_name << ": "
                 << fixed << setprecision(3) << ps.normalization_factor << "x" << endl;
        }
        cout << "=======================================" << endl;
    }

    return result;
}

// ============================================================================
// Gradient Computation
// ============================================================================

double SensitivityAnalyzer::compute_gradient(
    const EventCameraGeneticOptimizer::Genome& base_genome,
    ParameterIndex param_index,
    double delta,
    function<float(const EventCameraGeneticOptimizer::Genome&)> fitness_evaluator,
    double& fitness_plus,
    double& fitness_minus)
{
    if (config_.use_central_difference) {
        // Central difference: df/dx ≈ (f(x+h) - f(x-h)) / (2h)
        // More accurate than forward difference

        // Create genome with +delta perturbation
        EventCameraGeneticOptimizer::Genome genome_plus = base_genome;
        perturb_parameter(genome_plus, param_index, +delta);
        genome_plus.clamp();

        // Create genome with -delta perturbation
        EventCameraGeneticOptimizer::Genome genome_minus = base_genome;
        perturb_parameter(genome_minus, param_index, -delta);
        genome_minus.clamp();

        // Evaluate fitness at both points
        fitness_plus = fitness_evaluator(genome_plus);
        fitness_minus = fitness_evaluator(genome_minus);

        // Compute gradient (use absolute value for sensitivity)
        double gradient = abs(fitness_plus - fitness_minus) / (2.0 * delta);

        return gradient;
    } else {
        // Forward difference: df/dx ≈ (f(x+h) - f(x)) / h
        // Less accurate but only requires one additional evaluation

        EventCameraGeneticOptimizer::Genome genome_plus = base_genome;
        perturb_parameter(genome_plus, param_index, +delta);
        genome_plus.clamp();

        fitness_plus = fitness_evaluator(genome_plus);
        fitness_minus = fitness_evaluator(base_genome);

        double gradient = abs(fitness_plus - fitness_minus) / delta;

        return gradient;
    }
}

// ============================================================================
// Parameter Access Functions
// ============================================================================

void SensitivityAnalyzer::perturb_parameter(
    EventCameraGeneticOptimizer::Genome& genome,
    ParameterIndex param_index,
    double delta) const
{
    switch (param_index) {
        case BIAS_DIFF_ON:
            genome.bias_diff_on += static_cast<int>(delta);
            break;
        case BIAS_DIFF_OFF:
            genome.bias_diff_off += static_cast<int>(delta);
            break;
        case BIAS_HPF:
            genome.bias_hpf += static_cast<int>(delta);
            break;
        case TRAIL_THRESHOLD:
            genome.trail_threshold_us += static_cast<int>(delta);
            break;
        default:
            cerr << "ERROR: Unknown parameter index: " << param_index << endl;
            break;
    }
}

string SensitivityAnalyzer::get_parameter_name(ParameterIndex param_index) const
{
    switch (param_index) {
        case BIAS_DIFF_ON: return "bias_diff_on";
        case BIAS_DIFF_OFF: return "bias_diff_off";
        case BIAS_HPF: return "bias_hpf";
        case TRAIL_THRESHOLD: return "trail_threshold_us";
        default: return "unknown";
    }
}

double SensitivityAnalyzer::get_parameter_value(
    const EventCameraGeneticOptimizer::Genome& genome,
    ParameterIndex param_index) const
{
    switch (param_index) {
        case BIAS_DIFF_ON: return genome.bias_diff_on;
        case BIAS_DIFF_OFF: return genome.bias_diff_off;
        case BIAS_HPF: return genome.bias_hpf;
        case TRAIL_THRESHOLD: return genome.trail_threshold_us;
        default: return 0.0;
    }
}

double SensitivityAnalyzer::get_parameter_range(
    const EventCameraGeneticOptimizer::Genome& genome,
    ParameterIndex param_index) const
{
    switch (param_index) {
        case BIAS_DIFF_ON:
            return genome.ranges.diff_on_max - genome.ranges.diff_on_min;
        case BIAS_DIFF_OFF:
            return genome.ranges.diff_off_max - genome.ranges.diff_off_min;
        case BIAS_HPF:
            return genome.ranges.hpf_max - genome.ranges.hpf_min;
        case TRAIL_THRESHOLD:
            return genome.ranges.trail_max - genome.ranges.trail_min;
        default:
            return 1.0;
    }
}

// ============================================================================
// Logging Functions
// ============================================================================

void SensitivityAnalyzer::log_result(const SensitivityResult& result)
{
    ofstream log_file;

    // Open file (append mode)
    log_file.open(config_.log_file, ios::app);

    if (!log_file.is_open()) {
        cerr << "ERROR: Could not open sensitivity log file: " << config_.log_file << endl;
        return;
    }

    // Write header if this is the first write
    if (!log_header_written_) {
        log_file << "generation,timestamp,parameter,base_value,sensitivity,delta,fitness_plus,fitness_minus,base_fitness" << endl;
        log_header_written_ = true;
    }

    // Format timestamp
    auto time_t = chrono::system_clock::to_time_t(result.timestamp);
    stringstream timestamp_ss;
    timestamp_ss << put_time(localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    string timestamp_str = timestamp_ss.str();

    // Write one row per parameter
    for (const auto& ps : result.param_sensitivities) {
        log_file << result.generation << ","
                << timestamp_str << ","
                << ps.param_name << ","
                << ps.base_value << ","
                << fixed << setprecision(8) << ps.sensitivity << ","
                << ps.delta_used << ","
                << ps.fitness_plus << ","
                << ps.fitness_minus << ","
                << result.base_fitness << endl;
    }

    log_file.close();
}

void SensitivityAnalyzer::print_result(const SensitivityResult& result) const
{
    cout << "\n=======================================" << endl;
    cout << "Sensitivity Analysis Results" << endl;
    cout << "Generation: " << result.generation << endl;
    cout << "=======================================" << endl;
    cout << "Base Fitness: " << result.base_fitness << endl;
    cout << "\nParameter Sensitivities:" << endl;
    cout << left << setw(20) << "Parameter"
         << right << setw(12) << "Base Value"
         << setw(15) << "Sensitivity"
         << setw(12) << "Rel. Impact" << endl;
    cout << string(59, '-') << endl;

    for (const auto& ps : result.param_sensitivities) {
        double relative_impact = (result.max_sensitivity > 0)
            ? (ps.sensitivity / result.max_sensitivity)
            : 0.0;

        cout << left << setw(20) << ps.param_name
             << right << setw(12) << fixed << setprecision(2) << ps.base_value
             << setw(15) << setprecision(6) << ps.sensitivity
             << setw(11) << setprecision(1) << (relative_impact * 100.0) << "%" << endl;
    }

    cout << "\nSensitivity Range:" << endl;
    cout << "  Max: " << result.max_sensitivity << endl;
    cout << "  Min: " << result.min_sensitivity << endl;
    cout << "  Ratio: " << (result.max_sensitivity / max(result.min_sensitivity, 1e-10)) << "x" << endl;
    cout << "=======================================" << endl;
}

// ============================================================================
// Normalization Factor Computation (Phase 2)
// ============================================================================

void SensitivityAnalyzer::compute_normalization_factors(SensitivityResult& result)
{
    // Find max sensitivity for normalization
    double max_sens = result.max_sensitivity;

    // Avoid division by zero
    if (max_sens < 1e-10) {
        max_sens = 1.0;
    }

    // Compute inverse scaling factors for each parameter
    for (auto& ps : result.param_sensitivities) {
        if (ps.sensitivity > 1e-10) {
            // Inverse relationship: high sensitivity = low factor = smaller mutations
            // If sensitivity = max_sens, factor = 1.0 (baseline)
            // If sensitivity = max_sens/2, factor = 2.0 (2x larger mutations)
            // If sensitivity = max_sens*2, factor = 0.5 (2x smaller mutations)
            ps.normalization_factor = max_sens / ps.sensitivity;

            // Clamp to prevent extreme values
            ps.normalization_factor = clamp(
                ps.normalization_factor,
                config_.min_normalization_factor,
                config_.max_normalization_factor
            );
        } else {
            // Zero or near-zero sensitivity: use baseline scaling
            ps.normalization_factor = 1.0;
        }
    }
}
