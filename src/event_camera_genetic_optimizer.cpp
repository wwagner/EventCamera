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
    bias_refr = 0;
    bias_fo = 0;
    bias_hpf = 0;
    bias_pr = 0;
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
    uniform_int_distribution<int> refr_dist(ranges.refr_min, ranges.refr_max);
    uniform_int_distribution<int> fo_dist(ranges.fo_min, ranges.fo_max);
    uniform_int_distribution<int> hpf_dist(ranges.hpf_min, ranges.hpf_max);
    uniform_int_distribution<int> pr_dist(ranges.pr_min, ranges.pr_max);

    bias_diff = diff_dist(rng);
    bias_refr = refr_dist(rng);
    bias_fo = fo_dist(rng);
    bias_hpf = hpf_dist(rng);
    bias_pr = pr_dist(rng);

    // Randomize accumulation time (log-scale for better distribution)
    uniform_real_distribution<float> log_accum_dist(log(0.001f), log(0.1f));
    accumulation_time_s = exp(log_accum_dist(rng));

    // Randomize trail filter
    bernoulli_distribution enable_dist(0.3);  // 30% chance to enable
    enable_trail_filter = enable_dist(rng);
    uniform_int_distribution<int> trail_dist(ranges.trail_min, ranges.trail_max);
    trail_threshold_us = trail_dist(rng);

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
    bias_refr = max(ranges.refr_min, min(ranges.refr_max, bias_refr));
    bias_fo = max(ranges.fo_min, min(ranges.fo_max, bias_fo));
    bias_hpf = max(ranges.hpf_min, min(ranges.hpf_max, bias_hpf));
    bias_pr = max(ranges.pr_min, min(ranges.pr_max, bias_pr));

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
        result.combined_fitness = params_.alpha * (1.0f / result.contrast_score)
                                + params_.beta * result.noise_metric;
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
        new_population.push_back(population_[indices[i]]);
        new_fitness.push_back(fitness_cache_[indices[i]]);

        if (fitness_cache_[indices[i]].combined_fitness < best_fitness_) {
            best_fitness_ = fitness_cache_[indices[i]].combined_fitness;
            best_genome_ = population_[indices[i]];
        }
    }

    // Fill rest with offspring
    while (new_population.size() < static_cast<size_t>(params_.population_size)) {
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
    offspring.bias_pr = coin_flip(rng_) ? parent1.bias_pr : parent2.bias_pr;

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
    if (mutate_dist(rng_)) {
        int range = genome.ranges.pr_max - genome.ranges.pr_min;
        normal_distribution<float> noise(0.0f, params_.mutation_strength * range);
        genome.bias_pr += static_cast<int>(noise(rng_));
    }

    // Mutate accumulation time (log-space)
    if (mutate_dist(rng_)) {
        float log_val = log(genome.accumulation_time_s);
        normal_distribution<float> noise(0.0f, params_.mutation_strength * (log(0.1f) - log(0.001f)));
        log_val += noise(rng_);
        genome.accumulation_time_s = exp(log_val);
    }

    // Mutate trail filter
    if (mutate_dist(rng_)) {
        genome.enable_trail_filter = !genome.enable_trail_filter;
    }
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
    file << "bias_refr = " << genome.bias_refr << endl;
    file << "bias_fo = " << genome.bias_fo << endl;
    file << "bias_hpf = " << genome.bias_hpf << endl;
    file << "bias_pr = " << genome.bias_pr << endl;
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
