#pragma once

#include <string>
#include <opencv2/core.hpp>

/**
 * AppConfig - Application configuration management
 *
 * Manages all configurable parameters for the event camera tracking application:
 * - Camera hardware settings (biases, accumulation time)
 * - Stereo calibration parameters (baseline, angles, elevation)
 * - 3D view settings (azimuth, elevation, distance)
 * - Window geometry
 * - Rendering parameters (colors, fonts, sizes)
 * - Algorithm parameters (thresholds, debug intervals)
 *
 * Configuration is loaded from event_config.ini in the working directory.
 *
 * NAMING CONVENTION:
 * - _m = meters
 * - _p = pixels
 * - _d = degrees
 * - _s = seconds
 * - _us = microseconds
 * - _rgb = RGB color (0-255, 0-255, 0-255)
 * - No suffix = unitless (ratios, counts, flags)
 */
class AppConfig {
public:
    // Camera hardware settings
    struct CameraSettings {
        int bias_diff = 128;        // Event detection threshold (0-255)
        int bias_refr = 128;        // Refractory period (0-255)
        int bias_fo = 128;          // Photoreceptor follower (0-255)
        int bias_hpf = 128;         // High-pass filter (0-255)
        int bias_pr = 128;          // Pixel photoreceptor (0-255)
        float accumulation_time_s = 0.01f;  // Event accumulation period (0.001-0.05 seconds)
    };

    // Stereo calibration settings
    struct StereoSettings {
        float baseline_m = 0.20f;      // Camera separation (meters) - 20cm apart
        float yaw_deg = 10.0f;         // Camera yaw angle (degrees) - cameras angled inward
        float pitch_deg = 0.0f;        // Left camera pitch angle (degrees)
        float camera_elevation_m = 0.0f;  // Height above ground (meters)
        float fov_horizontal_deg = 74.5f;  // Horizontal field of view (degrees)
        float fov_vertical_deg = 43.5f;    // Vertical field of view (degrees)
    };

    // 3D view settings
    struct ViewSettings {
        double azimuth = 180.0;      // View azimuth angle (degrees) - behind view
        double elevation = -35.0;    // View elevation angle (degrees) - looking down
        double distance = 1.8;       // Camera distance from origin (meters)
    };

    // LED detection settings
    struct LEDDetectionSettings {
        int min_brightness = 5;         // Minimum brightness threshold (0-255)
        float min_area_m2 = 0.000002f;  // Minimum LED area (square meters) ~2px at 1000px/m
        float max_area_m2 = 0.002f;     // Maximum LED area (square meters) ~2000px at 1000px/m
    };

    // Stereo matching settings
    struct StereoMatchingSettings {
        float max_y_diff_m = 0.20f;     // Maximum vertical disparity (meters) ~200px at 1000px/m
        float max_dist_m = 0.50f;       // Maximum matching distance (meters) ~500px at 1000px/m
    };

    // Triangulation filter settings
    struct TriangulationSettings {
        float min_depth_m = 0.001f;     // Minimum valid depth (0.0001-1.0 meters)
        float max_depth_m = 1.0f;       // Maximum valid depth (0.1-100.0 meters)
        float max_distance_m = 1.0f;    // Maximum distance from origin (0.1-100.0 meters)
        float min_w_threshold = 1e-8f;  // Minimum w value for homogeneous coordinates (1e-10 to 1e-6)
        bool enable_distortion_correction = true; // Apply lens distortion correction before triangulation
        float radial_bias_correction = 0.0f;  // Radial bias correction factor (-0.5 to 0.5)
                                               // Positive values push points away from camera convergence point
                                               // Negative values pull points toward convergence point
        float x_offset_m = 0.0f;        // X-axis offset bias in meters (-0.5 to 0.5)
        float y_offset_m = 0.0f;        // Y-axis offset bias in meters (-0.5 to 0.5)
        float z_offset_m = 0.0f;        // Z-axis offset bias in meters (-0.5 to 0.5)
    };

    // Tracking algorithm settings
    struct TrackingSettings {
        int min_leds_for_tracking = 4;      // Minimum LEDs needed for pose estimation (3-10)
        float scale_correction_factor = 8.8f; // Legacy parameter (superseded by coordinate transformation)
        float max_radius_error_ratio = 0.5f;  // Maximum radius error tolerance (0.1-2.0, currently 50%)
        float min_confidence = 0.3f;          // Minimum tracking confidence (0.0-1.0, currently 30%)
        float outlier_distance_multiplier = 3.0f; // Outlier rejection threshold (median distance multiplier, 2.0-5.0)
        float outlier_tolerance_m = 0.05f;    // Additional tolerance for outlier rejection (meters)
    };

    // Cluster-based tracking settings (NEW)
    struct ClusterTrackingSettings {
        int min_leds_per_cluster = 2;        // Minimum LEDs to recognize cluster (2-3)
        int min_clusters_for_tracking = 2;   // Minimum clusters needed for pose (2-4)
        float max_cluster_match_distance_m = 0.015f;  // Maximum distance to match LEDs to cluster (meters)
        bool use_color_validation = true;    // Validate RGB pattern in clusters
        float color_tolerance = 0.3f;        // Color matching tolerance (0.0-1.0)
    };

    // Window geometry
    struct WindowSettings {
        int width = 1280;            // 2 columns x 640 pixels
        int height = 840;            // Top row 480 + bottom row 360 pixels
    };

    // Simulation viewer settings
    struct SimulationSettings {
        int window_width = 1920;           // Simulation window width (0 = fullscreen)
        int window_height = 1080;          // Simulation window height (0 = fullscreen)
        bool fullscreen = false;           // Start in fullscreen mode
        int num_objects = 1;               // Number of tracked objects
        float sphere_diameter_m = 0.15f;   // Sphere diameter in meters (default 0.15m = 15cm)
        // LED cluster configuration (NEW)
        bool use_led_clusters = true;      // Enable cluster mode (true) or single LED mode (false)
        int cluster_count = 12;            // Number of LED clusters on sphere
        int leds_per_cluster = 3;          // LEDs per cluster (typically 3 for RGB)
        float cluster_separation_m = 0.005f; // Distance between LEDs in cluster (meters)
        // Legacy single LED configuration
        int led_count = 12;                // Number of LEDs per sphere (used when use_led_clusters=false)
        float led_radius_m = 0.003f;       // LED marker radius in meters (default 0.003m = 3mm)
        float distance_from_screen_m = 0.5f; // Distance from screen to sphere(s) in meters
    };

    // Debug output settings
    struct DebugSettings {
        int verbosity = 2;  // Debug verbosity level:
                            // 0 = Silent (no debug output)
                            // 1 = Errors only (critical failures)
                            // 2 = Warnings + Important info (tracking results, config)
                            // 3 = Info (periodic updates every ~30-60 frames)
                            // 4 = Verbose (frame-by-frame details, all matrices)

        bool show_projection_matrices = false;     // Show projection matrices (verbosity >= 4)
        bool show_triangulation_details = false;   // Show per-point triangulation (verbosity >= 4)
        bool show_pose_estimation = false;         // Show pose estimation details (verbosity >= 3)
        bool show_coordinate_transform = false;    // Show coordinate transformations (verbosity >= 4)
    };

    // Rendering settings - 3D view elements
    struct Rendering3DViewSettings {
        cv::Scalar background_color_rgb = cv::Scalar(30, 30, 30);  // Dark gray background
        cv::Scalar grid_color_rgb = cv::Scalar(80, 80, 80);        // Grid line color
        float grid_size_m = 1.0f;                                   // Grid extends ±1m
        float grid_step_m = 0.1f;                                   // Grid line spacing (10cm)

        // Axes rendering
        float axes_font_scale = 0.5f;                               // Text size for axis labels
        int axes_line_thickness = 2;                                // Thickness for axis lines
        float axes_length_multiplier = 2.0f;                        // Make axes 2x sphere radius
        int axes_arrow_thickness = 3;                               // Axis arrow line width
        float axes_arrow_tip_ratio = 0.15f;                         // Arrowhead size relative to length
    };

    // Rendering settings - camera visualization
    struct RenderingCameraSettings {
        cv::Scalar left_frustum_color_rgb = cv::Scalar(0, 255, 255);   // Cyan for left camera
        cv::Scalar right_frustum_color_rgb = cv::Scalar(255, 255, 0);  // Yellow for right camera
        cv::Scalar left_fov_color_rgb = cv::Scalar(255, 200, 0);       // Blue-cyan for left FOV
        cv::Scalar right_fov_color_rgb = cv::Scalar(0, 200, 255);      // Orange-yellow for right FOV

        float frustum_size_m = 0.3f;                                    // Size of camera box
        float frustum_size_small_m = 0.05f;                             // Smaller frustum variant
        int frustum_line_thickness = 2;                                 // Edge thickness for frustums
        int fov_line_thickness = 1;                                     // Thin lines for FOV boundaries
        float fov_plane_alpha = 0.15f;                                  // Transparency for FOV planes
        float fov_plane_near_dist_m = 0.1f;                             // FOV plane start distance

        // FOV plane labels
        float fov_label_font_scale = 0.8f;                              // Font size for FOV labels (1-8)
        int fov_label_thickness = 2;                                    // Thickness for FOV labels
        float fov_label_distance_ratio = 0.25f;                         // Label position (25% of max extent)
    };

    // Rendering settings - ground truth visualization
    struct RenderingGroundTruthSettings {
        cv::Scalar sphere_fill_color_rgb = cv::Scalar(200, 100, 0);    // Blue filled sphere
        cv::Scalar sphere_outline_color_rgb = cv::Scalar(255, 150, 0); // Blue outline
        int sphere_outline_thickness = 2;                               // Sphere edge thickness

        int led_radius_p = 4;                                           // Filled LED size in pixels
        int led_outline_radius_p = 6;                                   // LED glow size in pixels
        cv::Scalar led_fill_color_rgb = cv::Scalar(0, 0, 255);         // Red filled LEDs
        cv::Scalar led_outline_color_rgb = cv::Scalar(0, 0, 150);      // Dark red outline
        int led_outline_thickness = 1;                                  // LED edge thickness
    };

    // Rendering settings - estimated pose visualization
    struct RenderingEstimatedSettings {
        cv::Scalar sphere_outline_color_rgb = cv::Scalar(0, 255, 0);   // Green outline
        int sphere_outline_thickness = 3;                               // Thick green line
        float sphere_inner_ratio = 0.8f;                                // Inner circle at 80% radius
        cv::Scalar sphere_inner_color_rgb = cv::Scalar(0, 200, 0);     // Lighter green
        int sphere_inner_thickness = 2;                                 // Inner circle line width
        float min_confidence = 0.3f;                                    // Show if confidence > 0.3
    };

    // Rendering settings - text and labels
    struct RenderingTextSettings {
        float label_font_scale = 0.5f;                                  // Text size for labels
        int label_thickness = 2;                                        // Thickness for labels
        cv::Scalar label_background_color_rgb = cv::Scalar(0, 0, 0);   // Black background

        // Tracking info overlay
        float tracking_info_font_scale = 0.6f;                          // Tracking overlay text size
        int tracking_info_thickness = 2;                                // Bold overlay text
        cv::Scalar tracking_info_color_rgb = cv::Scalar(0, 255, 0);    // Green text

        // Error display colors
        cv::Scalar error_excellent_color_rgb = cv::Scalar(0, 255, 0);  // Green for error < 1mm
        cv::Scalar tracking_lost_color_rgb = cv::Scalar(255, 0, 0);    // Red for lost tracking
    };

    // Algorithm tuning parameters
    struct AlgorithmSettings {
        int periodic_debug_interval = 60;                              // Periodic debug every N frames
        int triangulation_debug_interval = 30;                         // Triangulation debug every N frames
        int pose_debug_interval = 30;                                  // Pose estimation debug every N frames
        int debug_match_limit = 5;                                     // Maximum debug matches to log
    };

    // Genetic Algorithm settings
    struct GeneticAlgorithmSettings {
        int population_size = 30;           // Number of genomes per generation
        int num_generations = 20;           // Number of generations to evolve
        float mutation_rate = 0.15f;        // Probability of mutating each gene
        float crossover_rate = 0.7f;        // Probability of crossover vs cloning
        int frames_per_eval = 30;           // Frames to capture per fitness evaluation

        // Parameters to optimize (0=fixed, 1=optimize)
        bool optimize_bias_diff = true;
        bool optimize_bias_refr = true;
        bool optimize_bias_fo = true;
        bool optimize_bias_hpf = true;
        bool optimize_bias_pr = true;
        bool optimize_accumulation = true;
        bool optimize_trail_filter = false;
        bool optimize_antiflicker = false;
        bool optimize_erc = false;
    };

    AppConfig();
    ~AppConfig();

    // Load configuration from file (returns true if successful)
    // Default path points to root directory (../../../ from build/bin/Release)
    bool load(const std::string& filename = "../../../event_config.ini");

    // Accessors
    CameraSettings& camera_settings() { return camera_settings_; }
    const CameraSettings& camera_settings() const { return camera_settings_; }

    StereoSettings& stereo_settings() { return stereo_settings_; }
    const StereoSettings& stereo_settings() const { return stereo_settings_; }

    ViewSettings& view_settings() { return view_settings_; }
    const ViewSettings& view_settings() const { return view_settings_; }

    LEDDetectionSettings& led_detection_settings() { return led_detection_settings_; }
    const LEDDetectionSettings& led_detection_settings() const { return led_detection_settings_; }

    StereoMatchingSettings& stereo_matching_settings() { return stereo_matching_settings_; }
    const StereoMatchingSettings& stereo_matching_settings() const { return stereo_matching_settings_; }

    TriangulationSettings& triangulation_settings() { return triangulation_settings_; }
    const TriangulationSettings& triangulation_settings() const { return triangulation_settings_; }

    TrackingSettings& tracking_settings() { return tracking_settings_; }
    const TrackingSettings& tracking_settings() const { return tracking_settings_; }

    ClusterTrackingSettings& cluster_tracking_settings() { return cluster_tracking_settings_; }
    const ClusterTrackingSettings& cluster_tracking_settings() const { return cluster_tracking_settings_; }

    WindowSettings& window_settings() { return window_settings_; }
    const WindowSettings& window_settings() const { return window_settings_; }

    SimulationSettings& simulation_settings() { return simulation_settings_; }
    const SimulationSettings& simulation_settings() const { return simulation_settings_; }

    DebugSettings& debug_settings() { return debug_settings_; }
    const DebugSettings& debug_settings() const { return debug_settings_; }

    Rendering3DViewSettings& rendering_3d_view() { return rendering_3d_view_; }
    const Rendering3DViewSettings& rendering_3d_view() const { return rendering_3d_view_; }

    RenderingCameraSettings& rendering_camera() { return rendering_camera_; }
    const RenderingCameraSettings& rendering_camera() const { return rendering_camera_; }

    RenderingGroundTruthSettings& rendering_ground_truth() { return rendering_ground_truth_; }
    const RenderingGroundTruthSettings& rendering_ground_truth() const { return rendering_ground_truth_; }

    RenderingEstimatedSettings& rendering_estimated() { return rendering_estimated_; }
    const RenderingEstimatedSettings& rendering_estimated() const { return rendering_estimated_; }

    RenderingTextSettings& rendering_text() { return rendering_text_; }
    const RenderingTextSettings& rendering_text() const { return rendering_text_; }

    AlgorithmSettings& algorithm_settings() { return algorithm_settings_; }
    const AlgorithmSettings& algorithm_settings() const { return algorithm_settings_; }

    GeneticAlgorithmSettings& ga_settings() { return ga_settings_; }
    const GeneticAlgorithmSettings& ga_settings() const { return ga_settings_; }

private:
    // Helper function to parse RGB color from "R, G, B" string
    cv::Scalar parse_rgb(const std::string& value);

    CameraSettings camera_settings_;
    StereoSettings stereo_settings_;
    ViewSettings view_settings_;
    LEDDetectionSettings led_detection_settings_;
    StereoMatchingSettings stereo_matching_settings_;
    TriangulationSettings triangulation_settings_;
    TrackingSettings tracking_settings_;
    ClusterTrackingSettings cluster_tracking_settings_;
    WindowSettings window_settings_;
    SimulationSettings simulation_settings_;
    DebugSettings debug_settings_;
    Rendering3DViewSettings rendering_3d_view_;
    RenderingCameraSettings rendering_camera_;
    RenderingGroundTruthSettings rendering_ground_truth_;
    RenderingEstimatedSettings rendering_estimated_;
    RenderingTextSettings rendering_text_;
    AlgorithmSettings algorithm_settings_;
    GeneticAlgorithmSettings ga_settings_;
};
