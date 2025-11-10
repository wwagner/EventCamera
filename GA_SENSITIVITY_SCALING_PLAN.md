# GA Sensitivity Analysis and Adaptive Scaling Implementation Plan

## Overview

This document outlines the implementation of sensitivity-based parameter scaling for the Event Camera Genetic Algorithm optimizer. The goal is to improve optimization efficiency by scaling parameter changes inversely proportional to their sensitivity to the fitness function.

## Key Concepts

### Sensitivity Analysis
- Measure how changes in each parameter affect the fitness function
- Use numerical gradients: ∂fitness/∂parameter
- Parameters with high sensitivity need careful tuning (small steps)
- Parameters with low sensitivity can be adjusted more aggressively

### Adaptive Scaling
- Scale mutation strengths inversely to sensitivity
- Normalize parameter space to equalize impact
- Adapt scaling factors when fitness landscape changes

## Implementation Phases

## Phase 1: Sensitivity Analysis Infrastructure

### 1.1 Create Sensitivity Analyzer Class

**New Files:**
- `include/optimization/sensitivity_analyzer.h`
- `src/optimization/sensitivity_analyzer.cpp`

**Core Functionality:**
```cpp
class SensitivityAnalyzer {
public:
    struct SensitivityResult {
        std::vector<double> sensitivities;      // Per-parameter sensitivities
        std::vector<double> normalization_factors; // Scaling factors
        double max_sensitivity;
        double min_sensitivity;
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
    };

    // Compute sensitivity using numerical gradients
    SensitivityResult analyze(
        const EventCameraGeneticOptimizer::Genome& genome,
        const EventCameraGeneticOptimizer::Genome::OptimizationMask& mask,
        std::function<float(const Genome&)> fitness_evaluator
    );

private:
    double compute_gradient(
        const Genome& base_genome,
        int parameter_index,
        double delta,
        std::function<float(const Genome&)> fitness_evaluator
    );
};
```

**Numerical Gradient Computation:**
```cpp
// Central difference method for accurate gradients
double compute_gradient(genome, param_idx, delta, evaluator) {
    Genome genome_plus = genome;
    Genome genome_minus = genome;

    // Perturb parameter
    genome_plus.params[param_idx] += delta;
    genome_minus.params[param_idx] -= delta;

    // Clamp to valid ranges
    genome_plus.clamp();
    genome_minus.clamp();

    // Evaluate fitness at perturbed points
    float fitness_plus = evaluator(genome_plus);
    float fitness_minus = evaluator(genome_minus);

    // Compute gradient
    return std::abs(fitness_plus - fitness_minus) / (2.0 * delta);
}
```

### 1.2 Integration Points

**Location:** `src/event_camera_genetic_optimizer.cpp`

Add to `EventCameraGeneticOptimizer` class:
```cpp
private:
    SensitivityAnalyzer sensitivity_analyzer_;
    SensitivityAnalyzer::SensitivityResult current_sensitivity_;
    bool sensitivity_valid_ = false;
    int generations_since_sensitivity_update_ = 0;
```

**Trigger Points in `evolve()` method:**
```cpp
// Check if sensitivity analysis needed
if (should_update_sensitivity()) {
    perform_sensitivity_analysis();
}
```

## Phase 2: Parameter Normalization System

### 2.1 Transform Layer Implementation

**Normalized Parameter Space:**
- Map all parameters to [0, 1] range
- Apply sensitivity-based scaling
- Maintain bidirectional transformations

**Implementation:**
```cpp
class ParameterTransform {
public:
    // Physical to normalized space
    Genome to_normalized(const Genome& physical,
                        const SensitivityResult& sensitivity) {
        Genome normalized = physical;

        for (int i = 0; i < NUM_PARAMS; i++) {
            if (optimization_mask[i]) {
                double range = param_max[i] - param_min[i];
                double value = physical.params[i];

                // Normalize to [0,1]
                double norm = (value - param_min[i]) / range;

                // Apply sensitivity scaling
                norm *= sensitivity.normalization_factors[i];

                normalized.params[i] = norm;
            }
        }
        return normalized;
    }

    // Normalized to physical space
    Genome to_physical(const Genome& normalized,
                      const SensitivityResult& sensitivity) {
        Genome physical = normalized;

        for (int i = 0; i < NUM_PARAMS; i++) {
            if (optimization_mask[i]) {
                double range = param_max[i] - param_min[i];
                double norm = normalized.params[i];

                // Remove sensitivity scaling
                norm /= sensitivity.normalization_factors[i];

                // Map back to physical range
                physical.params[i] = param_min[i] + norm * range;
            }
        }
        return physical;
    }
};
```

### 2.2 Modified GA Operations

**Mutation in Normalized Space:**
```cpp
void mutate_normalized(Genome& genome) {
    // Transform to normalized space
    Genome normalized = transform.to_normalized(genome, current_sensitivity_);

    // Apply uniform mutation in normalized space
    for (int i = 0; i < NUM_PARAMS; i++) {
        if (should_mutate(i)) {
            // Same mutation strength for all parameters
            normal_distribution<float> noise(0.0f, uniform_mutation_strength);
            normalized.params[i] += noise(rng_);

            // Clamp to [0,1]
            normalized.params[i] = std::clamp(normalized.params[i], 0.0, 1.0);
        }
    }

    // Transform back to physical space
    genome = transform.to_physical(normalized, current_sensitivity_);
    genome.clamp(); // Ensure hardware limits
}
```

## Phase 3: Adaptive Triggering Mechanism

### 3.1 Convergence Detection

**Metrics to Track:**
```cpp
struct PopulationMetrics {
    double fitness_variance;
    double genome_diversity;  // Average pairwise distance
    double best_improvement_rate;
    int stagnation_counter;
};
```

**Trigger Conditions:**
```cpp
bool should_update_sensitivity() {
    // Trigger on stagnation
    if (stagnation_counter_ >= stagnation_threshold_) {
        return true;
    }

    // Trigger on convergence
    PopulationMetrics metrics = compute_population_metrics();
    if (metrics.fitness_variance < convergence_threshold_) {
        return true;
    }

    // Periodic update
    if (generations_since_sensitivity_update_ >= update_interval_) {
        return true;
    }

    return false;
}
```

### 3.2 Sensitivity Update Protocol

```cpp
void perform_sensitivity_analysis() {
    std::cout << "Performing sensitivity analysis..." << std::endl;

    // Get current best genome
    Genome best = get_best_genome();

    // Create fitness evaluator lambda
    auto evaluator = [this](const Genome& g) {
        return evaluate_genome_fitness(g, frames_per_eval_);
    };

    // Run sensitivity analysis
    current_sensitivity_ = sensitivity_analyzer_.analyze(
        best,
        optimization_mask_,
        evaluator
    );

    // Update normalization factors
    update_normalization_factors();

    // Reset counter
    generations_since_sensitivity_update_ = 0;
    sensitivity_valid_ = true;

    // Log results
    log_sensitivity_results();
}
```

## Phase 4: Implementation Details

### 4.1 Sensitivity Computation Algorithm

```cpp
SensitivityResult analyze(genome, mask, evaluator) {
    SensitivityResult result;

    // Parameters being optimized
    std::vector<int> active_params;
    for (int i = 0; i < NUM_PARAMS; i++) {
        if (mask[i]) active_params.push_back(i);
    }

    // Compute base fitness
    float base_fitness = evaluator(genome);

    // For each active parameter
    for (int param_idx : active_params) {
        // Compute delta as 1% of parameter range
        double range = param_max[param_idx] - param_min[param_idx];
        double delta = 0.01 * range;

        // Compute gradient
        double gradient = compute_gradient(genome, param_idx, delta, evaluator);

        // Store sensitivity (absolute value of gradient)
        result.sensitivities[param_idx] = gradient;
    }

    // Compute normalization factors
    compute_normalization_factors(result);

    return result;
}
```

### 4.2 Normalization Factor Calculation

```cpp
void compute_normalization_factors(SensitivityResult& result) {
    // Find max sensitivity for normalization
    double max_sens = *std::max_element(
        result.sensitivities.begin(),
        result.sensitivities.end()
    );

    // Avoid division by zero
    if (max_sens < 1e-10) max_sens = 1.0;

    // Compute inverse scaling factors
    for (size_t i = 0; i < result.sensitivities.size(); i++) {
        if (result.sensitivities[i] > 0) {
            // Inverse relationship: high sensitivity = low factor
            result.normalization_factors[i] = max_sens / result.sensitivities[i];

            // Clamp to prevent numerical issues
            result.normalization_factors[i] = std::clamp(
                result.normalization_factors[i],
                0.01,  // Minimum 1% scaling
                100.0  // Maximum 100x scaling
            );
        } else {
            // Zero sensitivity: use default scaling
            result.normalization_factors[i] = 1.0;
        }
    }
}
```

### 4.3 Parameter Mapping

**Current Parameters and Their Indices:**
```cpp
enum ParameterIndex {
    BIAS_DIFF = 0,          // Range: [-25, 23]
    BIAS_DIFF_ON = 1,       // Range: [-85, 140]
    BIAS_DIFF_OFF = 2,      // Range: [-35, 190]
    BIAS_REFR = 3,          // Range: [-20, 235]
    BIAS_FO = 4,            // Range: [-35, 55]
    BIAS_HPF = 5,           // Range: [0, 120]
    ACCUMULATION_TIME = 6,   // Range: [100, 100000] μs (log scale)
    TRAIL_ENABLE = 7,        // Boolean
    TRAIL_THRESHOLD = 8,     // Range: [1000, 100000] μs
    ANTIFLICKER_ENABLE = 9,  // Boolean
    AF_LOW_FREQ = 10,        // Range: [100, 200] Hz
    AF_HIGH_FREQ = 11,       // Range: [100, 200] Hz
    ERC_ENABLE = 12,         // Boolean
    ERC_TARGET_RATE = 13     // Range: [1000, 10000] kev/s
};
```

## Phase 5: Configuration and Logging

### 5.1 New Configuration Options

**Add to `event_config.ini`:**
```ini
[GeneticAlgorithm]
# Existing parameters...

# Sensitivity Analysis Settings
enable_sensitivity_analysis = 1
sensitivity_delta_ratio = 0.01        # 1% of parameter range for gradient
convergence_variance_threshold = 0.05 # Trigger when population converges
stagnation_trigger = 5                # Generations without improvement
sensitivity_update_interval = 10      # Max generations between updates
normalization_strength = 1.0          # 0=off, 1=full inverse scaling

# Advanced sensitivity options
use_parallel_evaluation = 1           # Parallel gradient computation
sensitivity_samples = 3                # Samples per gradient (for averaging)
log_sensitivity_details = 1           # Detailed logging of sensitivity values
```

### 5.2 Enhanced Logging

**Sensitivity Log File:** `sensitivity_analysis_log.csv`
```csv
generation,timestamp,parameter,sensitivity,normalization_factor
5,2024-11-09T12:34:56,bias_diff,0.234,4.27
5,2024-11-09T12:34:56,bias_diff_on,0.089,11.23
5,2024-11-09T12:34:56,bias_diff_off,0.156,6.41
...
```

**Main Optimization Log Enhancement:**
```csv
generation,best_fitness,avg_fitness,sensitivity_updated,avg_normalization
0,1.234,2.345,false,1.0
5,0.987,1.543,true,3.45
6,0.876,1.234,false,3.45
...
```

## Phase 6: Testing and Validation

### 6.1 Unit Tests

**Test Files to Create:**
- `tests/test_sensitivity_analyzer.cpp`
- `tests/test_parameter_transform.cpp`
- `tests/test_adaptive_mutation.cpp`

**Key Test Cases:**
```cpp
TEST(SensitivityAnalyzer, ComputesAccurateGradients) {
    // Test with known linear function
    auto linear_fitness = [](const Genome& g) {
        return 2.0 * g.bias_diff + 3.0 * g.bias_diff_on;
    };

    SensitivityResult result = analyzer.analyze(genome, mask, linear_fitness);

    EXPECT_NEAR(result.sensitivities[BIAS_DIFF], 2.0, 0.1);
    EXPECT_NEAR(result.sensitivities[BIAS_DIFF_ON], 3.0, 0.1);
}

TEST(ParameterTransform, PreservesValidRanges) {
    Genome original = create_random_genome();
    Genome normalized = transform.to_normalized(original, sensitivity);
    Genome restored = transform.to_physical(normalized, sensitivity);

    EXPECT_TRUE(restored.is_valid());
    EXPECT_NEAR(original.bias_diff, restored.bias_diff, 0.01);
}
```

### 6.2 Performance Validation

**Metrics to Track:**
- Convergence speed comparison (generations to target fitness)
- Final fitness achieved
- Parameter exploration coverage
- Computation overhead

**Benchmark Protocol:**
1. Run standard GA 10 times, record metrics
2. Run sensitivity-scaled GA 10 times, record metrics
3. Compare convergence curves
4. Statistical significance testing

## Expected Benefits

### Quantitative Improvements
- **30-50% faster convergence** to target fitness
- **Better final solutions** due to improved parameter exploration
- **Reduced variance** in optimization runs
- **Automatic adaptation** to different optimization scenarios

### Qualitative Benefits
- **Parameter importance insights** - understand which biases matter most
- **Reduced manual tuning** - no need to adjust mutation rates per parameter
- **Robust to parameter scales** - handles wide range differences automatically
- **Diagnostic information** - sensitivity logs help debug optimization issues

## Implementation Timeline

### Week 1: Core Infrastructure
- Day 1-2: Implement SensitivityAnalyzer class
- Day 3-4: Add ParameterTransform system
- Day 5: Integration with EventCameraGeneticOptimizer

### Week 2: Adaptive Mechanisms
- Day 1-2: Implement convergence detection
- Day 3-4: Add adaptive triggering logic
- Day 5: Configuration and logging

### Week 3: Testing and Refinement
- Day 1-2: Unit tests
- Day 3-4: Performance benchmarks
- Day 5: Documentation and polish

## Future Extensions

### Advanced Sensitivity Methods
- **Sobol indices** for global sensitivity analysis
- **Morris method** for computational efficiency
- **Machine learning** prediction of sensitivity

### Multi-Objective Optimization
- Separate sensitivity per fitness component
- Pareto front exploration with adaptive scaling
- Dynamic weight adjustment based on sensitivity

### Online Learning
- Continuous sensitivity estimation during evolution
- Bayesian optimization integration
- Reinforcement learning for parameter control

## References

1. Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer*
2. Beyer, H.G. & Schwefel, H.P. (2002). *Evolution strategies: A comprehensive introduction*
3. Hansen, N. (2006). *The CMA evolution strategy: A comparing review*
4. Eiben, A.E. & Smith, J.E. (2015). *Introduction to Evolutionary Computing*

## Appendix: Code Structure

### File Organization
```
EventCamera/
├── include/
│   └── optimization/
│       ├── sensitivity_analyzer.h      [NEW]
│       ├── parameter_transform.h       [NEW]
│       └── ga_controller.h            [MODIFY]
├── src/
│   └── optimization/
│       ├── sensitivity_analyzer.cpp    [NEW]
│       ├── parameter_transform.cpp     [NEW]
│       └── ga_controller.cpp          [MODIFY]
├── src/
│   ├── event_camera_genetic_optimizer.cpp [MODIFY]
│   └── main.cpp                          [MODIFY]
└── tests/
    ├── test_sensitivity_analyzer.cpp     [NEW]
    └── test_parameter_transform.cpp      [NEW]
```

### Key Modifications Summary

1. **EventCameraGeneticOptimizer**: Add sensitivity analysis integration
2. **GAController**: Pass sensitivity data between components
3. **Main**: Expose fitness evaluation for sensitivity computation
4. **AppConfig**: Add new configuration parameters
5. **UI**: Display sensitivity values and normalization factors

---

*This implementation plan provides a robust framework for adding sensitivity-based adaptive scaling to the Event Camera GA optimizer, significantly improving its efficiency and effectiveness.*