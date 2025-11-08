#pragma once

#include <vector>
#include <random>
#include <functional>
#include <string>
#include <atomic>
#include <opencv2/core.hpp>

/**
 * EventCameraGeneticOptimizer - Genetic algorithm for optimizing event camera parameters
 *
 * Optimizes camera biases and filter settings to maximize contrast and minimize noise.
 *
 * Fitness Function:
 *   fitness = alpha * (1.0 / contrast) + beta * noise_metric
 *
 * Where:
 *   contrast = standard deviation of frame intensities (higher is better)
 *   noise_metric = temporal variance + spatial noise (lower is better)
 *
 * Lower fitness is better (minimization problem).
 */
class EventCameraGeneticOptimizer {
public:
    /**
     * Parameter genome - encodes all optimizable event camera parameters
     */
    struct Genome {
        // Camera Biases (6 params) - ranges will be set from actual hardware
        int bias_diff;              // Event detection threshold (global)
        int bias_diff_on;           // ON event detection threshold
        int bias_diff_off;          // OFF event detection threshold
        int bias_refr;              // Refractory period
        int bias_fo;                // Photoreceptor follower
        int bias_hpf;               // High-pass filter

        // Frame Generation (1 param)
        float accumulation_time_s;  // 0.001-0.1 seconds

        // Event Trail Filter (2 params)
        bool enable_trail_filter;   // Enable/disable
        int trail_threshold_us;     // Threshold in microseconds

        // Anti-Flicker (3 params)
        bool enable_antiflicker;    // Enable/disable
        int af_low_freq;            // Low frequency (Hz)
        int af_high_freq;           // High frequency (Hz)

        // Event Rate Controller (2 params)
        bool enable_erc;            // Enable/disable
        int erc_target_rate;        // Target event rate (kev/s)

        // Bias ranges (set from camera hardware at runtime)
        struct BiasRanges {
            int diff_min = -25, diff_max = 23;
            int diff_on_min = -25, diff_on_max = 23;
            int diff_off_min = -25, diff_off_max = 23;
            int refr_min = -25, refr_max = 23;
            int fo_min = -25, fo_max = 23;
            int hpf_min = -25, hpf_max = 23;
            int pr_min = -25, pr_max = 23;
            int trail_min = 1000, trail_max = 100000;
            int af_freq_min = 100, af_freq_max = 200;
            int erc_rate_min = 1000, erc_rate_max = 10000;
        } ranges;

        // Optimization mask (which parameters to optimize)
        struct OptimizationMask {
            bool bias_diff = true;
            bool bias_diff_on = true;
            bool bias_diff_off = true;
            bool bias_refr = true;
            bool bias_fo = true;
            bool bias_hpf = true;
            bool accumulation = true;
            bool trail_filter = false;
            bool antiflicker = false;
            bool erc = false;
        } opt_mask;

        Genome();  // Initialize with default values
        void randomize(std::mt19937& rng);  // Randomize within bounds (respects opt_mask)
        void clamp();  // Enforce bounds and constraints
        void set_ranges(const BiasRanges& r);  // Set hardware-specific ranges
    };

    /**
     * Fitness result for a genome
     */
    struct FitnessResult {
        float contrast_score;       // Frame contrast (std dev of intensities)
        float noise_metric;         // Noise measure (temporal + spatial variance)
        float combined_fitness;     // alpha * (1/contrast) + beta * noise
        int num_valid_frames;       // Number of frames captured
        int total_frames;           // Total frames attempted

        // Additional metrics
        float temporal_variance;    // Frame-to-frame variance
        float spatial_noise;        // Within-frame noise estimate
        float mean_brightness;      // Average frame brightness
        float event_rate_avg;       // Average event rate (kev/s)
        float event_rate_std;       // Event rate stability
        float isolated_pixel_ratio; // Ratio of isolated pixels (bad noise)
        float cluster_fill_metric;  // Cluster fill quality (0.0 = perfect fill, 1.0 = many gaps)
        int total_event_pixels;     // Total number of bright pixels (events)

        FitnessResult() : contrast_score(0.0f), noise_metric(1e6f),
                         combined_fitness(1e6f), num_valid_frames(0),
                         total_frames(0), temporal_variance(1e6f),
                         spatial_noise(1e6f), mean_brightness(0.0f),
                         event_rate_avg(0.0f), event_rate_std(1e6f),
                         isolated_pixel_ratio(1e6f), cluster_fill_metric(1e6f),
                         total_event_pixels(0) {}
    };

    /**
     * Fitness evaluation callback
     * User provides a function that:
     * 1. Applies genome parameters to camera
     * 2. Captures N frames
     * 3. Measures contrast and noise
     * 4. Returns FitnessResult
     */
    using FitnessCallback = std::function<FitnessResult(const Genome&)>;

    /**
     * Progress callback for GUI updates
     */
    using ProgressCallback = std::function<void(int generation, float best_fitness,
                                                const Genome& best_genome,
                                                const FitnessResult& best_result)>;

    /**
     * Optimization parameters
     */
    struct OptimizerParams {
        int population_size = 30;           // Number of genomes per generation
        int num_generations = 50;           // Number of generations to evolve
        float mutation_rate = 0.15f;        // Probability of mutating each gene
        float mutation_strength = 0.5f;     // Magnitude of mutations (fraction of range) - INCREASED for faster convergence
        float crossover_rate = 0.7f;        // Probability of crossover vs cloning
        float elite_fraction = 0.1f;        // Fraction of top performers to preserve

        // Fitness weights
        float alpha = 1.0f;                 // Weight for contrast (want high contrast)
        float beta = 0.5f;                  // Weight for noise (want low noise)
        float gamma = 2.0f;                 // Weight for isolated pixels (want clusters, not noise)
        float epsilon = 3.0f;               // Weight for cluster fill (want bridging pixels, no gaps)

        // Event count constraint
        int minimum_event_pixels = 500;     // Minimum bright pixels required (23 dots × ~25-50 px each)
        float delta = 5.0f;                 // Penalty weight for insufficient events

        // Stopping criteria
        float target_fitness = 0.01f;       // Stop if fitness below this
        int stagnation_limit = 15;          // Stop if no improvement for N generations

        // Logging
        int log_interval = 1;               // Log every N generations
        std::string log_file = "event_camera_optimization_log.csv";
        std::string best_config_file = "best_event_config.ini";

        // Optimization mask (which parameters to optimize)
        Genome::OptimizationMask opt_mask;
    };

    /**
     * Constructor
     * @param params Optimizer parameters
     * @param fitness_callback Function to evaluate genome fitness
     * @param progress_callback Optional callback for GUI updates
     */
    EventCameraGeneticOptimizer(const OptimizerParams& params,
                                FitnessCallback fitness_callback,
                                ProgressCallback progress_callback = nullptr);

    /**
     * Run optimization (blocking)
     * @return Best genome found
     */
    Genome optimize();

    /**
     * Stop optimization (thread-safe)
     */
    void stop();

    /**
     * Get current generation number
     */
    int get_generation() const { return generation_; }

    /**
     * Get best fitness so far
     */
    float get_best_fitness() const { return best_fitness_; }

    /**
     * Get best genome so far
     */
    const Genome& get_best_genome() const { return best_genome_; }

    /**
     * Check if optimization is running
     */
    bool is_running() const { return running_; }

    /**
     * Save genome to INI file
     */
    static void save_genome_to_ini(const Genome& genome, const std::string& filename);

    /**
     * Load genome from INI file
     */
    static Genome load_genome_from_ini(const std::string& filename);

    /**
     * Calculate contrast score from frame
     */
    static float calculate_contrast(const cv::Mat& frame);

    /**
     * Calculate noise metric from multiple frames
     */
    static float calculate_noise(const std::vector<cv::Mat>& frames);

    /**
     * Calculate temporal variance across frames
     */
    static float calculate_temporal_variance(const std::vector<cv::Mat>& frames);

    /**
     * Calculate spatial noise within a frame
     */
    static float calculate_spatial_noise(const cv::Mat& frame);

    /**
     * Calculate ratio of isolated pixels (single-pixel noise) vs clustered pixels
     * Returns value between 0.0 (all clustered) and 1.0 (all isolated)
     * @param frame Input frame
     * @param min_cluster_radius Minimum radius for cluster detection (default 2 pixels)
     */
    static float calculate_isolated_pixels(const cv::Mat& frame, int min_cluster_radius = 2);

    /**
     * Calculate cluster fill metric - measures how well pixels bridge together
     * Returns value between 0.0 (perfect fill, no gaps) and 1.0 (many gaps)
     * @param frame Input frame
     * @param min_cluster_radius Minimum radius for cluster operations (default 2 pixels)
     */
    static float calculate_cluster_fill(const cv::Mat& frame, int min_cluster_radius = 2);

    /**
     * Calculate connected component fitness metric
     * Finds connected pixel groups and evaluates how well they match target size
     * @param frame Input frame
     * @param target_radius Target radius for connected components (pixels)
     * @param min_cluster_radius Minimum radius for noise filtering (default 2)
     * @return Fitness value (lower is better): deviation from target + noise penalty
     */
    static float calculate_connected_component_fitness(const cv::Mat& frame,
                                                        int target_radius,
                                                        int min_cluster_radius = 2);

    /**
     * Calculate cluster-based fitness metric (DEPRECATED - use calculate_connected_component_fitness)
     * Evaluates events within defined circular clusters and penalizes noise outside
     * @param frame Input frame
     * @param cluster_centers Vector of (x,y) cluster center positions
     * @param cluster_radius Radius of each circular cluster
     * @param min_cluster_radius Minimum radius for morphological operations
     * @return Fitness value (lower is better): ratio of events outside clusters + isolated pixel ratio
     */
    static float calculate_cluster_fitness(const cv::Mat& frame,
                                           const std::vector<std::pair<int, int>>& cluster_centers,
                                           int cluster_radius,
                                           int min_cluster_radius = 2);

private:
    OptimizerParams params_;
    FitnessCallback fitness_callback_;
    ProgressCallback progress_callback_;
    std::mt19937 rng_;
    std::atomic<bool> running_;
    std::atomic<bool> should_stop_;

    // Evolution state
    int generation_;
    std::vector<Genome> population_;
    std::vector<FitnessResult> fitness_cache_;
    Genome best_genome_;
    float best_fitness_;
    int stagnation_counter_;

    // Statistics
    std::vector<float> best_fitness_history_;
    std::vector<float> avg_fitness_history_;

    // Helper methods
    void initialize_population();
    FitnessResult evaluate_fitness(const Genome& genome);
    void selection_and_reproduction();
    Genome crossover(const Genome& parent1, const Genome& parent2);
    void mutate(Genome& genome);
    int tournament_selection(int tournament_size = 3);
    void log_generation();
    void save_best_genome();
    void notify_progress();
};
