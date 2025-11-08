#include "event_camera_genetic_optimizer.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <opencv2/imgproc.hpp>

using namespace std;

// ============================================================================
// Genome Implementation
// ============================================================================

EventCameraGeneticOptimizer::Genome::Genome() {
    // Initialize with reasonable defaults
    bias_diff = 0;
    bias_diff_on = 0;
    bias_diff_off = 0;
    bias_refr = 0;
    bias_fo = 0;
    bias_hpf = 0;
    accumulation_time_s = 0.01f;
    enable_trail_filter = false;
    trail_threshold_us = 10000;
    enable_antiflicker = false;
    af_low_freq = 100;
    af_high_freq = 150;
    enable_erc = false;
    erc_target_rate = 5000;
}

void EventCameraGeneticOptimizer::Genome::randomize(mt19937& rng) {
    // Randomize camera biases
    uniform_int_distribution<int> diff_dist(ranges.diff_min, ranges.diff_max);
    uniform_int_distribution<int> diff_on_dist(ranges.diff_on_min, ranges.diff_on_max);
    uniform_int_distribution<int> diff_off_dist(ranges.diff_off_min, ranges.diff_off_max);
    uniform_int_distribution<int> refr_dist(ranges.refr_min, ranges.refr_max);
    uniform_int_distribution<int> fo_dist(ranges.fo_min, ranges.fo_max);
    uniform_int_distribution<int> hpf_dist(ranges.hpf_min, ranges.hpf_max);

    bias_diff = diff_dist(rng);
    bias_diff_on = diff_on_dist(rng);
    bias_diff_off = diff_off_dist(rng);
    bias_refr = refr_dist(rng);
    bias_fo = fo_dist(rng);
    bias_hpf = hpf_dist(rng);

    // Randomize accumulation time (log-scale for better distribution)
    uniform_real_distribution<float> log_accum_dist(log(0.001f), log(0.1f));
    accumulation_time_s = exp(log_accum_dist(rng));

    // Trail filter: Always enabled when optimizing, only randomize threshold
    enable_trail_filter = true;  // Always ON - filter type will be STC_KEEP_TRAIL
    uniform_int_distribution<int> trail_dist(ranges.trail_min, ranges.trail_max);
    trail_threshold_us = trail_dist(rng);

    // Boolean distribution for optional features
    bernoulli_distribution enable_dist(0.5);

    // Randomize anti-flicker
    enable_antiflicker = enable_dist(rng);
    uniform_int_distribution<int> freq_dist(ranges.af_freq_min, ranges.af_freq_max);
    af_low_freq = freq_dist(rng);
    af_high_freq = af_low_freq + 10 + (freq_dist(rng) % 50);  // Ensure high > low

    // Randomize ERC
    enable_erc = enable_dist(rng);
    uniform_int_distribution<int> erc_dist(ranges.erc_rate_min, ranges.erc_rate_max);
    erc_target_rate = erc_dist(rng);

    clamp();
}

void EventCameraGeneticOptimizer::Genome::clamp() {
    // Clamp camera biases
    bias_diff = max(ranges.diff_min, min(ranges.diff_max, bias_diff));
    bias_diff_on = max(ranges.diff_on_min, min(ranges.diff_on_max, bias_diff_on));
    bias_diff_off = max(ranges.diff_off_min, min(ranges.diff_off_max, bias_diff_off));
    bias_refr = max(ranges.refr_min, min(ranges.refr_max, bias_refr));
    bias_fo = max(ranges.fo_min, min(ranges.fo_max, bias_fo));
    bias_hpf = max(ranges.hpf_min, min(ranges.hpf_max, bias_hpf));

    // Clamp accumulation time
    accumulation_time_s = max(0.001f, min(0.1f, accumulation_time_s));

    // Clamp trail filter
    trail_threshold_us = max(ranges.trail_min, min(ranges.trail_max, trail_threshold_us));

    // Clamp anti-flicker
    af_low_freq = max(ranges.af_freq_min, min(ranges.af_freq_max, af_low_freq));
    af_high_freq = max(ranges.af_freq_min, min(ranges.af_freq_max, af_high_freq));
    if (af_low_freq >= af_high_freq) {
        af_high_freq = af_low_freq + 10;
    }

    // Clamp ERC
    erc_target_rate = max(ranges.erc_rate_min, min(ranges.erc_rate_max, erc_target_rate));
}

void EventCameraGeneticOptimizer::Genome::set_ranges(const BiasRanges& r) {
    ranges = r;
}

// ============================================================================
// EventCameraGeneticOptimizer Implementation
// ============================================================================

EventCameraGeneticOptimizer::EventCameraGeneticOptimizer(
    const OptimizerParams& params,
    FitnessCallback fitness_callback,
    ProgressCallback progress_callback)
    : params_(params)
    , fitness_callback_(fitness_callback)
    , progress_callback_(progress_callback)
    , rng_(random_device{}())
    , running_(false)
    , should_stop_(false)
    , generation_(0)
    , best_fitness_(1e9f)
    , stagnation_counter_(0)
{
    population_.resize(params_.population_size);
    fitness_cache_.resize(params_.population_size);
}

EventCameraGeneticOptimizer::Genome EventCameraGeneticOptimizer::optimize() {
    running_ = true;
    should_stop_ = false;
    generation_ = 0;
    stagnation_counter_ = 0;

    cout << "=======================================" << endl;
    cout << "Event Camera Genetic Optimization" << endl;
    cout << "=======================================" << endl;
    cout << "Population: " << params_.population_size << endl;
    cout << "Generations: " << params_.num_generations << endl;
    cout << "Mutation rate: " << params_.mutation_rate << endl;
    cout << "Crossover rate: " << params_.crossover_rate << endl;
    cout << "=======================================" << endl;

    // Initialize population
    initialize_population();

    // Open log file
    ofstream log_file(params_.log_file);
    log_file << "generation,best_fitness,avg_fitness,best_contrast,best_noise,valid_frames" << endl;

    // Evolution loop
    for (generation_ = 0; generation_ < params_.num_generations; ++generation_) {
        if (should_stop_) {
            cout << "Optimization stopped by user" << endl;
            break;
        }

        // Selection and reproduction
        selection_and_reproduction();

        // Log progress
        if (generation_ % params_.log_interval == 0) {
            log_generation();

            // Write to log file
            float avg_fitness = 0.0f;
            for (const auto& fit : fitness_cache_) {
                avg_fitness += fit.combined_fitness;
            }
            avg_fitness /= fitness_cache_.size();

            log_file << generation_ << ","
                    << best_fitness_ << ","
                    << avg_fitness << ","
                    << evaluate_fitness(best_genome_).contrast_score << ","
                    << evaluate_fitness(best_genome_).noise_metric << ","
                    << evaluate_fitness(best_genome_).num_valid_frames << endl;
        }

        // Notify progress callback
        if (progress_callback_) {
            notify_progress();
        }

        // Check stopping criteria
        if (best_fitness_ < params_.target_fitness) {
            cout << "Target fitness reached!" << endl;
            break;
        }

        if (stagnation_counter_ >= params_.stagnation_limit) {
            cout << "Stagnation limit reached. No improvement for "
                 << stagnation_counter_ << " generations." << endl;
            break;
        }
    }

    log_file.close();

    // Save best genome
    save_best_genome();

    cout << "=======================================" << endl;
    cout << "Optimization Complete!" << endl;
    cout << "Best fitness: " << best_fitness_ << endl;
    cout << "Best config saved to: " << params_.best_config_file << endl;
    cout << "=======================================" << endl;

    running_ = false;
    return best_genome_;
}

void EventCameraGeneticOptimizer::stop() {
    should_stop_ = true;
}

void EventCameraGeneticOptimizer::initialize_population() {
    cout << "Initializing population..." << endl;

    for (int i = 0; i < params_.population_size; ++i) {
        // Check for stop request
        if (should_stop_) {
            cout << "Population initialization stopped by user" << endl;
            break;
        }

        population_[i].randomize(rng_);
        fitness_cache_[i] = evaluate_fitness(population_[i]);

        // Update best
        if (fitness_cache_[i].combined_fitness < best_fitness_) {
            best_fitness_ = fitness_cache_[i].combined_fitness;
            best_genome_ = population_[i];
        }
    }

    cout << "Initial best fitness: " << best_fitness_ << endl;
}

EventCameraGeneticOptimizer::FitnessResult
EventCameraGeneticOptimizer::evaluate_fitness(const Genome& genome) {
    // Call user-provided fitness callback
    FitnessResult result = fitness_callback_(genome);

    // Calculate combined fitness
    if (result.contrast_score > 0.0f) {
        // Calculate event count penalty (penalize if too few events)
        float event_penalty = 0.0f;
        if (result.total_event_pixels < params_.minimum_event_pixels) {
            float deficit_ratio = static_cast<float>(params_.minimum_event_pixels - result.total_event_pixels)
                                / static_cast<float>(params_.minimum_event_pixels);
            event_penalty = params_.delta * deficit_ratio;
        }

        result.combined_fitness = params_.alpha * (1.0f / result.contrast_score)
                                + params_.beta * result.noise_metric
                                + params_.gamma * result.isolated_pixel_ratio
                                + params_.epsilon * result.cluster_fill_metric
                                + event_penalty;
    } else {
        result.combined_fitness = 1e9f;  // Invalid
    }

    return result;
}

void EventCameraGeneticOptimizer::selection_and_reproduction() {
    vector<Genome> new_population;
    vector<FitnessResult> new_fitness;

    // Elitism - preserve top performers
    int num_elite = static_cast<int>(params_.population_size * params_.elite_fraction);

    // Sort by fitness
    vector<int> indices(params_.population_size);
    for (int i = 0; i < params_.population_size; ++i) {
        indices[i] = i;
    }
    sort(indices.begin(), indices.end(), [this](int a, int b) {
        return fitness_cache_[a].combined_fitness < fitness_cache_[b].combined_fitness;
    });

    // Add elite
    float previous_best = best_fitness_;
    for (int i = 0; i < num_elite; ++i) {
        // Check for stop request
        if (should_stop_) {
            break;
        }

        new_population.push_back(population_[indices[i]]);
        new_fitness.push_back(fitness_cache_[indices[i]]);

        if (fitness_cache_[indices[i]].combined_fitness < best_fitness_) {
            best_fitness_ = fitness_cache_[indices[i]].combined_fitness;
            best_genome_ = population_[indices[i]];
        }
    }

    // Fill rest with offspring
    while (new_population.size() < static_cast<size_t>(params_.population_size)) {
        // Check for stop request
        if (should_stop_) {
            cout << "Reproduction stopped by user" << endl;
            break;
        }

        // Select parents
        int parent1_idx = tournament_selection();
        int parent2_idx = tournament_selection();

        // Crossover or clone
        Genome offspring;
        bernoulli_distribution crossover_dist(params_.crossover_rate);
        if (crossover_dist(rng_)) {
            offspring = crossover(population_[parent1_idx], population_[parent2_idx]);
        } else {
            offspring = population_[parent1_idx];
        }

        // Mutate
        mutate(offspring);

        // Evaluate
        FitnessResult fit = evaluate_fitness(offspring);

        new_population.push_back(offspring);
        new_fitness.push_back(fit);

        // Update best
        if (fit.combined_fitness < best_fitness_) {
            best_fitness_ = fit.combined_fitness;
            best_genome_ = offspring;
        }
    }

    // Check for stagnation
    if (best_fitness_ >= previous_best - 1e-6f) {
        ++stagnation_counter_;
    } else {
        stagnation_counter_ = 0;
    }

    population_ = move(new_population);
    fitness_cache_ = move(new_fitness);
}

EventCameraGeneticOptimizer::Genome
EventCameraGeneticOptimizer::crossover(const Genome& parent1, const Genome& parent2) {
    Genome offspring;
    bernoulli_distribution coin_flip(0.5);

    // Uniform crossover - randomly pick each gene from either parent
    offspring.bias_diff = coin_flip(rng_) ? parent1.bias_diff : parent2.bias_diff;
    offspring.bias_refr = coin_flip(rng_) ? parent1.bias_refr : parent2.bias_refr;
    offspring.bias_fo = coin_flip(rng_) ? parent1.bias_fo : parent2.bias_fo;
    offspring.bias_hpf = coin_flip(rng_) ? parent1.bias_hpf : parent2.bias_hpf;

    offspring.accumulation_time_s = coin_flip(rng_) ?
        parent1.accumulation_time_s : parent2.accumulation_time_s;

    offspring.enable_trail_filter = coin_flip(rng_) ?
        parent1.enable_trail_filter : parent2.enable_trail_filter;
    offspring.trail_threshold_us = coin_flip(rng_) ?
        parent1.trail_threshold_us : parent2.trail_threshold_us;

    offspring.enable_antiflicker = coin_flip(rng_) ?
        parent1.enable_antiflicker : parent2.enable_antiflicker;
    offspring.af_low_freq = coin_flip(rng_) ? parent1.af_low_freq : parent2.af_low_freq;
    offspring.af_high_freq = coin_flip(rng_) ? parent1.af_high_freq : parent2.af_high_freq;

    offspring.enable_erc = coin_flip(rng_) ? parent1.enable_erc : parent2.enable_erc;
    offspring.erc_target_rate = coin_flip(rng_) ?
        parent1.erc_target_rate : parent2.erc_target_rate;

    offspring.ranges = parent1.ranges;  // Copy ranges
    offspring.clamp();

    return offspring;
}

void EventCameraGeneticOptimizer::mutate(Genome& genome) {
    bernoulli_distribution mutate_dist(params_.mutation_rate);

    // Mutate camera biases
    if (mutate_dist(rng_)) {
        int range = genome.ranges.diff_max - genome.ranges.diff_min;
        normal_distribution<float> noise(0.0f, params_.mutation_strength * range);
        genome.bias_diff += static_cast<int>(noise(rng_));
    }
    if (mutate_dist(rng_)) {
        int range = genome.ranges.diff_on_max - genome.ranges.diff_on_min;
        normal_distribution<float> noise(0.0f, params_.mutation_strength * range);
        genome.bias_diff_on += static_cast<int>(noise(rng_));
    }
    if (mutate_dist(rng_)) {
        int range = genome.ranges.diff_off_max - genome.ranges.diff_off_min;
        normal_distribution<float> noise(0.0f, params_.mutation_strength * range);
        genome.bias_diff_off += static_cast<int>(noise(rng_));
    }
    if (mutate_dist(rng_)) {
        int range = genome.ranges.refr_max - genome.ranges.refr_min;
        normal_distribution<float> noise(0.0f, params_.mutation_strength * range);
        genome.bias_refr += static_cast<int>(noise(rng_));
    }
    if (mutate_dist(rng_)) {
        int range = genome.ranges.fo_max - genome.ranges.fo_min;
        normal_distribution<float> noise(0.0f, params_.mutation_strength * range);
        genome.bias_fo += static_cast<int>(noise(rng_));
    }
    if (mutate_dist(rng_)) {
        int range = genome.ranges.hpf_max - genome.ranges.hpf_min;
        normal_distribution<float> noise(0.0f, params_.mutation_strength * range);
        genome.bias_hpf += static_cast<int>(noise(rng_));
    }

    // Mutate accumulation time (log-space)
    if (mutate_dist(rng_)) {
        float log_val = log(genome.accumulation_time_s);
        normal_distribution<float> noise(0.0f, params_.mutation_strength * (log(0.1f) - log(0.001f)));
        log_val += noise(rng_);
        genome.accumulation_time_s = exp(log_val);
    }

    // Mutate trail filter threshold only (enable is always true, type is STC_KEEP_TRAIL)
    if (mutate_dist(rng_)) {
        int range = genome.ranges.trail_max - genome.ranges.trail_min;
        normal_distribution<float> noise(0.0f, params_.mutation_strength * range);
        genome.trail_threshold_us += static_cast<int>(noise(rng_));
    }

    // Mutate anti-flicker
    if (mutate_dist(rng_)) {
        genome.enable_antiflicker = !genome.enable_antiflicker;
    }
    if (mutate_dist(rng_)) {
        int range = genome.ranges.af_freq_max - genome.ranges.af_freq_min;
        normal_distribution<float> noise(0.0f, params_.mutation_strength * range);
        genome.af_low_freq += static_cast<int>(noise(rng_));
        genome.af_high_freq += static_cast<int>(noise(rng_));
    }

    // Mutate ERC
    if (mutate_dist(rng_)) {
        genome.enable_erc = !genome.enable_erc;
    }
    if (mutate_dist(rng_)) {
        int range = genome.ranges.erc_rate_max - genome.ranges.erc_rate_min;
        normal_distribution<float> noise(0.0f, params_.mutation_strength * range);
        genome.erc_target_rate += static_cast<int>(noise(rng_));
    }

    genome.clamp();
}

int EventCameraGeneticOptimizer::tournament_selection(int tournament_size) {
    uniform_int_distribution<int> index_dist(0, params_.population_size - 1);

    int best_idx = index_dist(rng_);
    float best_fitness = fitness_cache_[best_idx].combined_fitness;

    for (int i = 1; i < tournament_size; ++i) {
        int idx = index_dist(rng_);
        if (fitness_cache_[idx].combined_fitness < best_fitness) {
            best_fitness = fitness_cache_[idx].combined_fitness;
            best_idx = idx;
        }
    }

    return best_idx;
}

void EventCameraGeneticOptimizer::log_generation() {
    FitnessResult best_result = evaluate_fitness(best_genome_);

    // Find worst genome in current population
    float worst_fitness = -1e9f;
    int worst_idx = 0;
    for (size_t i = 0; i < fitness_cache_.size(); ++i) {
        if (fitness_cache_[i].combined_fitness > worst_fitness) {
            worst_fitness = fitness_cache_[i].combined_fitness;
            worst_idx = i;
        }
    }
    const Genome& worst_genome = population_[worst_idx];
    FitnessResult worst_result = fitness_cache_[worst_idx];

    // Print best genome parameters
    cout << "  BEST:  diff=" << setw(3) << best_genome_.bias_diff
         << " diff_on=" << setw(3) << best_genome_.bias_diff_on
         << " diff_off=" << setw(3) << best_genome_.bias_diff_off
         << " refr=" << setw(3) << best_genome_.bias_refr
         << " fo=" << setw(3) << best_genome_.bias_fo
         << " hpf=" << setw(3) << best_genome_.bias_hpf
         << " | accum=" << fixed << setprecision(4) << best_genome_.accumulation_time_s
         << " | trail=" << (best_genome_.enable_trail_filter ? "ON " : "OFF")
         << (best_genome_.enable_trail_filter ? ("(" + to_string(best_genome_.trail_threshold_us) + "us)") : "     ")
         << " | af=" << (best_genome_.enable_antiflicker ? "ON " : "OFF")
         << (best_genome_.enable_antiflicker ? ("(" + to_string(best_genome_.af_low_freq) + "-" + to_string(best_genome_.af_high_freq) + "Hz)") : "        ")
         << " | erc=" << (best_genome_.enable_erc ? "ON " : "OFF")
         << (best_genome_.enable_erc ? ("(" + to_string(best_genome_.erc_target_rate) + "k)") : "     ")
         << endl;

    // Print worst genome parameters
    cout << "  WORST: diff=" << setw(3) << worst_genome.bias_diff
         << " diff_on=" << setw(3) << worst_genome.bias_diff_on
         << " diff_off=" << setw(3) << worst_genome.bias_diff_off
         << " refr=" << setw(3) << worst_genome.bias_refr
         << " fo=" << setw(3) << worst_genome.bias_fo
         << " hpf=" << setw(3) << worst_genome.bias_hpf
         << " | accum=" << fixed << setprecision(4) << worst_genome.accumulation_time_s
         << " | trail=" << (worst_genome.enable_trail_filter ? "ON " : "OFF")
         << (worst_genome.enable_trail_filter ? ("(" + to_string(worst_genome.trail_threshold_us) + "us)") : "     ")
         << " | af=" << (worst_genome.enable_antiflicker ? "ON " : "OFF")
         << (worst_genome.enable_antiflicker ? ("(" + to_string(worst_genome.af_low_freq) + "-" + to_string(worst_genome.af_high_freq) + "Hz)") : "        ")
         << " | erc=" << (worst_genome.enable_erc ? "ON " : "OFF")
         << (worst_genome.enable_erc ? ("(" + to_string(worst_genome.erc_target_rate) + "k)") : "     ")
         << endl;

    // Print generation summary
    cout << "Gen " << setw(4) << generation_
         << " | Fit: " << fixed << setprecision(6) << best_fitness_
         << " | Contrast: " << setprecision(2) << best_result.contrast_score
         << " | Noise: " << setprecision(4) << best_result.noise_metric
         << " | Valid: " << best_result.num_valid_frames << "/" << best_result.total_frames
         << " | Stag: " << stagnation_counter_
         << endl;
}

void EventCameraGeneticOptimizer::save_best_genome() {
    save_genome_to_ini(best_genome_, params_.best_config_file);
}

void EventCameraGeneticOptimizer::notify_progress() {
    FitnessResult best_result = evaluate_fitness(best_genome_);
    progress_callback_(generation_, best_fitness_, best_genome_, best_result);
}

// ============================================================================
// Static Utility Methods
// ============================================================================

void EventCameraGeneticOptimizer::save_genome_to_ini(const Genome& genome,
                                                     const string& filename) {
    ofstream file(filename);
    file << "# Best Event Camera Configuration" << endl;
    file << "# Generated by Genetic Algorithm Optimizer" << endl;
    file << endl;

    file << "[Camera]" << endl;
    file << "bias_diff = " << genome.bias_diff << endl;
    file << "bias_diff_on = " << genome.bias_diff_on << endl;
    file << "bias_diff_off = " << genome.bias_diff_off << endl;
    file << "bias_refr = " << genome.bias_refr << endl;
    file << "bias_fo = " << genome.bias_fo << endl;
    file << "bias_hpf = " << genome.bias_hpf << endl;
    file << "accumulation_time_s = " << genome.accumulation_time_s << endl;
    file << endl;

    file << "[EventTrailFilter]" << endl;
    file << "enable = " << (genome.enable_trail_filter ? "true" : "false") << endl;
    file << "threshold_us = " << genome.trail_threshold_us << endl;
    file << endl;

    file << "[AntiFlicker]" << endl;
    file << "enable = " << (genome.enable_antiflicker ? "true" : "false") << endl;
    file << "low_freq = " << genome.af_low_freq << endl;
    file << "high_freq = " << genome.af_high_freq << endl;
    file << endl;

    file << "[ERC]" << endl;
    file << "enable = " << (genome.enable_erc ? "true" : "false") << endl;
    file << "target_rate = " << genome.erc_target_rate << endl;

    file.close();
}

EventCameraGeneticOptimizer::Genome
EventCameraGeneticOptimizer::load_genome_from_ini(const string& filename) {
    // TODO: Implement INI parsing if needed
    Genome genome;
    return genome;
}

float EventCameraGeneticOptimizer::calculate_contrast(const cv::Mat& frame) {
    if (frame.empty()) return 0.0f;

    // Convert to grayscale if needed
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    // Calculate standard deviation as contrast measure
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);

    return static_cast<float>(stddev[0]);
}

float EventCameraGeneticOptimizer::calculate_noise(const vector<cv::Mat>& frames) {
    if (frames.size() < 2) return 1e6f;

    float temporal_var = calculate_temporal_variance(frames);
    float spatial_noise = calculate_spatial_noise(frames[0]);

    // Combined noise metric
    return temporal_var + spatial_noise;
}

float EventCameraGeneticOptimizer::calculate_temporal_variance(
    const vector<cv::Mat>& frames) {
    if (frames.size() < 2) return 1e6f;

    // Calculate frame-to-frame differences
    float total_diff = 0.0f;
    int count = 0;

    for (size_t i = 1; i < frames.size(); ++i) {
        cv::Mat diff;
        cv::absdiff(frames[i], frames[i-1], diff);

        cv::Scalar mean_diff = cv::mean(diff);
        total_diff += static_cast<float>(mean_diff[0]);
        ++count;
    }

    return count > 0 ? total_diff / count : 1e6f;
}

float EventCameraGeneticOptimizer::calculate_spatial_noise(const cv::Mat& frame) {
    if (frame.empty()) return 1e6f;

    // Use Laplacian variance as spatial noise estimate
    cv::Mat gray, laplacian;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    cv::Laplacian(gray, laplacian, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);

    // Lower variance in Laplacian = less noise
    return static_cast<float>(stddev[0]);
}

float EventCameraGeneticOptimizer::calculate_isolated_pixels(const cv::Mat& frame, int min_cluster_radius) {
    if (frame.empty()) return 1e6f;

    // Convert to grayscale if needed
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    // Threshold to find bright pixels (events)
    cv::Mat binary;
    cv::threshold(gray, binary, 10, 255, cv::THRESH_BINARY);

    // Count active pixels before erosion
    int pixels_before = cv::countNonZero(binary);
    if (pixels_before == 0) return 0.0f;  // No events, good

    // Erode to remove isolated pixels and small clusters using specified radius
    // Kernel size = 2*radius + 1 (e.g., radius=2 -> 5x5 kernel)
    int kernel_size = 2 * min_cluster_radius + 1;
    cv::Mat eroded;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::erode(binary, eroded, kernel);

    // Count active pixels after erosion (these are clustered)
    int pixels_after = cv::countNonZero(eroded);

    // Calculate ratio of isolated pixels (removed by erosion)
    int isolated_pixels = pixels_before - pixels_after;
    float isolated_ratio = static_cast<float>(isolated_pixels) / static_cast<float>(pixels_before);

    // Return ratio: 0.0 = all clustered (good), 1.0 = all isolated (bad)
    return isolated_ratio;
}

float EventCameraGeneticOptimizer::calculate_cluster_fill(const cv::Mat& frame, int min_cluster_radius) {
    if (frame.empty()) return 1e6f;

    // Convert to grayscale if needed
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    // Threshold to find bright pixels (events)
    cv::Mat binary;
    cv::threshold(gray, binary, 10, 255, cv::THRESH_BINARY);

    int pixels_before = cv::countNonZero(binary);
    if (pixels_before == 0) return 0.0f;  // No events, perfect fill

    // Apply closing operation to fill small gaps and connect nearby pixels
    // Closing = dilation followed by erosion - bridges small gaps
    int kernel_size = 2 * min_cluster_radius + 1;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::Mat closed;
    cv::morphologyEx(binary, closed, cv::MORPH_CLOSE, kernel);

    // Apply opening operation to remove noise while preserving structure
    // Opening = erosion followed by dilation - removes isolated pixels
    cv::Mat opened;
    cv::morphologyEx(closed, opened, cv::MORPH_OPEN, kernel);

    int pixels_after = cv::countNonZero(opened);

    // Calculate fill ratio: how many pixels remain after morphological ops
    // Higher ratio = better fill (pixels bridge together well)
    // Lower ratio = poor fill (many gaps and isolated pixels)
    float fill_ratio = (pixels_before > 0) ?
        static_cast<float>(pixels_after) / static_cast<float>(pixels_before) : 0.0f;

    // Return inverted (1 - ratio) so 0.0 = perfect fill, 1.0 = no fill
    return 1.0f - fill_ratio;
}

float EventCameraGeneticOptimizer::calculate_connected_component_fitness(
    const cv::Mat& frame,
    int target_radius,
    int min_cluster_radius) {

    if (frame.empty()) {
        return 1e6f;  // Invalid
    }

    // Convert to grayscale if needed
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    // Threshold to find bright pixels (events)
    cv::Mat binary;
    cv::threshold(gray, binary, 10, 255, cv::THRESH_BINARY);

    // Find connected components
    cv::Mat labels, stats, centroids;
    int num_components = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);

    // Calculate target area from target radius
    float target_area = CV_PI * target_radius * target_radius;

    float total_penalty = 0.0f;
    int num_valid_components = 0;
    int num_undersized = 0;
    float total_solidity_penalty = 0.0f;

    // Analyze each component (skip 0 which is background)
    for (int i = 1; i < num_components; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        // Calculate equivalent radius from area
        float equivalent_radius = std::sqrt(area / CV_PI);

        if (equivalent_radius < min_cluster_radius) {
            // Too small - likely noise, heavy penalty
            total_penalty += 10.0f;
            continue;
        }

        // Calculate solidity (compactness) of this component
        // Extract the component's pixels
        cv::Mat component_mask = (labels == i);

        // Find contours for this component
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(component_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        float solidity = 1.0f;  // Default for degenerate cases
        if (!contours.empty() && contours[0].size() >= 3) {
            // Calculate convex hull
            std::vector<cv::Point> hull;
            cv::convexHull(contours[0], hull);

            // Calculate areas
            double component_area = cv::contourArea(contours[0]);
            double hull_area = cv::contourArea(hull);

            if (hull_area > 0) {
                solidity = static_cast<float>(component_area / hull_area);
            }
        }

        // Penalize low solidity (noisy, sparse clusters)
        // Solid circles have solidity ~1.0, noisy clusters < 0.7
        float solidity_penalty = 0.0f;
        if (solidity < 0.8f) {
            // Heavy penalty for sparse/noisy clusters
            solidity_penalty = (0.8f - solidity) * 20.0f;  // 0-16 penalty
            total_solidity_penalty += solidity_penalty;
        }

        if (equivalent_radius < target_radius) {
            // Undersized component - penalize based on how much it's missing
            float deficit = target_radius - equivalent_radius;
            float deficit_ratio = deficit / target_radius;
            total_penalty += deficit_ratio * 5.0f + solidity_penalty;
            num_undersized++;
        } else {
            // Component meets or exceeds target
            num_valid_components++;

            // Reward solid, compact components more
            float reward = 0.5f;
            if (solidity > 0.9f) {
                reward += 1.0f;  // Extra reward for very solid components
            } else if (solidity > 0.8f) {
                reward += 0.5f;  // Moderate reward for fairly solid
            }

            total_penalty -= reward;

            // Add solidity penalty even for valid components
            total_penalty += solidity_penalty;

            // Small additional reward for being larger (encourages growth)
            float excess = equivalent_radius - target_radius;
            if (excess > 0) {
                total_penalty -= std::min(excess * 0.1f, 1.0f);  // Cap at 1.0 bonus
            }
        }
    }

    // Additional penalties
    if (num_valid_components == 0) {
        total_penalty += 50.0f;  // No valid components at all
    }

    // Add average solidity penalty to overall fitness
    // This emphasizes the importance of solid, filled dots
    total_penalty += total_solidity_penalty;

    // Return fitness (lower is better)
    return total_penalty;
}

float EventCameraGeneticOptimizer::calculate_cluster_fitness(
    const cv::Mat& frame,
    const std::vector<std::pair<int, int>>& cluster_centers,
    int cluster_radius,
    int min_cluster_radius) {

    if (frame.empty() || cluster_centers.empty()) {
        return 1e6f;  // Invalid
    }

    // Convert to grayscale if needed
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    // Threshold to find bright pixels (events)
    cv::Mat binary;
    cv::threshold(gray, binary, 10, 255, cv::THRESH_BINARY);

    // Apply morphological operations to remove isolated pixels
    int kernel_size = 2 * min_cluster_radius + 1;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::Mat filtered;
    cv::erode(binary, filtered, kernel);

    // Create mask for cluster regions (circular dots)
    cv::Mat cluster_mask = cv::Mat::zeros(gray.size(), CV_8U);
    for (const auto& center : cluster_centers) {
        cv::circle(cluster_mask, cv::Point(center.first, center.second),
                  cluster_radius, cv::Scalar(255), -1);  // Filled circle
    }

    // Calculate mean brightness inside cluster regions (using filtered events)
    cv::Scalar mean_inside = cv::mean(gray, cluster_mask);
    float brightness_inside = static_cast<float>(mean_inside[0]);

    // Count filtered events inside vs outside cluster regions
    cv::Mat events_inside_mask;
    cv::bitwise_and(filtered, cluster_mask, events_inside_mask);
    int events_inside = cv::countNonZero(events_inside_mask);

    cv::Mat events_outside_mask;
    cv::bitwise_and(filtered, ~cluster_mask, events_outside_mask);
    int events_outside = cv::countNonZero(events_outside_mask);

    // Calculate fill metric for overall frame
    float fill_metric = calculate_cluster_fill(frame, min_cluster_radius);

    // Calculate mean brightness outside cluster regions (background)
    cv::Mat outside_mask = ~cluster_mask;
    cv::Scalar mean_outside = cv::mean(gray, outside_mask);
    float brightness_outside = static_cast<float>(mean_outside[0]);

    // Calculate contrast: difference between inside and outside
    // Higher contrast = better (bright dots, dark background)
    float contrast = brightness_inside - brightness_outside;

    // Combined fitness metric (lower is better):
    // - Penalize low contrast (want bright clusters)
    // - Penalize events outside cluster regions (noise)
    // - Penalize poor fill (gaps and isolated pixels)
    // - Reward high background darkness
    float fitness = -contrast +
                   static_cast<float>(events_outside) * 0.1f +  // Penalize noise outside
                   fill_metric * 100.0f +                        // Penalize poor fill
                   brightness_outside;                           // Penalize bright background

    return fitness;
}
