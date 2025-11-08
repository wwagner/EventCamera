#include "app_config.h"
#include <fstream>
#include <sstream>
#include <iostream>

AppConfig::AppConfig() {
    // Constructor uses default values from struct definitions
}

AppConfig::~AppConfig() {
}

cv::Scalar AppConfig::parse_rgb(const std::string& value) {
    // Parse "R, G, B" format
    std::istringstream iss(value);
    int r, g, b;
    char comma;

    if (iss >> r >> comma >> g >> comma >> b) {
        // Clamp values to 0-255
        r = std::max(0, std::min(255, r));
        g = std::max(0, std::min(255, g));
        b = std::max(0, std::min(255, b));
        return cv::Scalar(b, g, r);  // OpenCV uses BGR format
    }

    // Return default gray if parsing fails
    return cv::Scalar(128, 128, 128);
}

bool AppConfig::load(const std::string& filename) {
    // Get absolute path for debugging
    #ifdef _WIN32
    char absolute_path[_MAX_PATH];
    _fullpath(absolute_path, filename.c_str(), _MAX_PATH);
    std::cout << "Attempting to load config from: " << absolute_path << std::endl;
    #else
    char absolute_path[PATH_MAX];
    realpath(filename.c_str(), absolute_path);
    std::cout << "Attempting to load config from: " << absolute_path << std::endl;
    #endif

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Config file not found: " << filename << " (using defaults)" << std::endl;
        std::cerr << "Note: Config file should be in the same directory as the executable" << std::endl;
        std::cerr << "  Using default camera bias settings (all 128)" << std::endl;
        std::cerr << "  Default frame accumulation time: " << camera_settings_.accumulation_time_s << "s" << std::endl;
        return false;
    }

    std::string line;
    std::string section;

    while (std::getline(file, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        size_t end = line.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) continue;  // Empty line
        line = line.substr(start, end - start + 1);

        // Skip comments
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;

        // Check for section header
        if (line[0] == '[' && line[line.length() - 1] == ']') {
            section = line.substr(1, line.length() - 2);
            continue;
        }

        // Parse key=value
        size_t equals = line.find('=');
        if (equals == std::string::npos) continue;

        std::string key = line.substr(0, equals);
        std::string value = line.substr(equals + 1);

        // Trim key and value
        key = key.substr(0, key.find_last_not_of(" \t") + 1);
        value = value.substr(value.find_first_not_of(" \t"));

        // Strip inline comments (anything after #)
        size_t comment_pos = value.find('#');
        if (comment_pos != std::string::npos) {
            value = value.substr(0, comment_pos);
        }
        // Trim trailing whitespace after removing comment
        size_t last_non_space = value.find_last_not_of(" \t");
        if (last_non_space != std::string::npos) {
            value = value.substr(0, last_non_space + 1);
        }

        // Parse based on section
        if (section == "Camera") {
            if (key == "bias_diff") camera_settings_.bias_diff = std::stoi(value);
            else if (key == "bias_diff_on") camera_settings_.bias_diff_on = std::stoi(value);
            else if (key == "bias_diff_off") camera_settings_.bias_diff_off = std::stoi(value);
            else if (key == "bias_fo") camera_settings_.bias_fo = std::stoi(value);
            else if (key == "bias_hpf") camera_settings_.bias_hpf = std::stoi(value);
            else if (key == "bias_refr") camera_settings_.bias_refr = std::stoi(value);
            else if (key == "accumulation_time_s") camera_settings_.accumulation_time_s = std::stof(value);
            // Backward compatibility: convert microseconds to seconds if old key is used
            else if (key == "accumulation_time_us") camera_settings_.accumulation_time_s = std::stof(value) / 1000000.0f;
            else if (key == "capture_directory") camera_settings_.capture_directory = value;
            // Trail Filter settings
            else if (key == "trail_filter_enabled") camera_settings_.trail_filter_enabled = (std::stoi(value) != 0);
            else if (key == "trail_filter_type") camera_settings_.trail_filter_type = std::stoi(value);
            else if (key == "trail_filter_threshold") camera_settings_.trail_filter_threshold = std::stoi(value);
            // ImageJ streaming settings
            else if (key == "imagej_streaming_enabled") camera_settings_.imagej_streaming_enabled = (std::stoi(value) != 0);
            else if (key == "imagej_stream_fps") camera_settings_.imagej_stream_fps = std::stoi(value);
            else if (key == "imagej_stream_directory") camera_settings_.imagej_stream_directory = value;
            else if (key == "imagej_max_stream_files") camera_settings_.imagej_max_stream_files = std::stoi(value);
        }
        else if (section == "Stereo") {
            if (key == "baseline_m") stereo_settings_.baseline_m = std::stof(value);
            // NEW naming convention: _d for degrees
            else if (key == "yaw_d") stereo_settings_.yaw_deg = std::stof(value);
            else if (key == "pitch_d") stereo_settings_.pitch_deg = std::stof(value);
            else if (key == "fov_horizontal_d") stereo_settings_.fov_horizontal_deg = std::stof(value);
            else if (key == "fov_vertical_d") stereo_settings_.fov_vertical_deg = std::stof(value);
            // BACKWARD COMPATIBILITY: old _deg suffix
            else if (key == "yaw_deg") stereo_settings_.yaw_deg = std::stof(value);
            else if (key == "pitch_deg") stereo_settings_.pitch_deg = std::stof(value);
            else if (key == "fov_horizontal_deg") stereo_settings_.fov_horizontal_deg = std::stof(value);
            else if (key == "fov_vertical_deg") stereo_settings_.fov_vertical_deg = std::stof(value);
            else if (key == "camera_elevation_m") stereo_settings_.camera_elevation_m = std::stof(value);
        }
        else if (section == "View") {
            // NEW naming convention: _d for degrees, _m for meters
            if (key == "azimuth_d") view_settings_.azimuth = std::stod(value);
            else if (key == "elevation_d") view_settings_.elevation = std::stod(value);
            else if (key == "distance_m") view_settings_.distance = std::stod(value);
            // BACKWARD COMPATIBILITY: old parameter names without suffix
            else if (key == "azimuth") view_settings_.azimuth = std::stod(value);
            else if (key == "elevation") view_settings_.elevation = std::stod(value);
            else if (key == "distance") view_settings_.distance = std::stod(value);
        }
        else if (section == "LEDDetection") {
            if (key == "min_brightness") led_detection_settings_.min_brightness = std::stoi(value);
            else if (key == "min_area_m2") led_detection_settings_.min_area_m2 = std::stof(value);
            else if (key == "max_area_m2") led_detection_settings_.max_area_m2 = std::stof(value);
        }
        else if (section == "StereoMatching") {
            if (key == "max_y_diff_m") stereo_matching_settings_.max_y_diff_m = std::stof(value);
            else if (key == "max_dist_m") stereo_matching_settings_.max_dist_m = std::stof(value);
        }
        else if (section == "Triangulation") {
            if (key == "min_depth_m") triangulation_settings_.min_depth_m = std::stof(value);
            else if (key == "max_depth_m") triangulation_settings_.max_depth_m = std::stof(value);
            else if (key == "max_distance_m") triangulation_settings_.max_distance_m = std::stof(value);
            else if (key == "min_w_threshold") triangulation_settings_.min_w_threshold = std::stof(value);
            else if (key == "enable_distortion_correction") triangulation_settings_.enable_distortion_correction = (std::stoi(value) != 0);
            else if (key == "radial_bias_correction") triangulation_settings_.radial_bias_correction = std::stof(value);
            else if (key == "x_offset_m") triangulation_settings_.x_offset_m = std::stof(value);
            else if (key == "y_offset_m") triangulation_settings_.y_offset_m = std::stof(value);
            else if (key == "z_offset_m") triangulation_settings_.z_offset_m = std::stof(value);
        }
        else if (section == "Tracking") {
            if (key == "min_leds_for_tracking") tracking_settings_.min_leds_for_tracking = std::stoi(value);
            else if (key == "scale_correction_factor") tracking_settings_.scale_correction_factor = std::stof(value);
            else if (key == "max_radius_error_ratio") tracking_settings_.max_radius_error_ratio = std::stof(value);
            else if (key == "min_confidence") tracking_settings_.min_confidence = std::stof(value);
            else if (key == "outlier_distance_multiplier") tracking_settings_.outlier_distance_multiplier = std::stof(value);
            else if (key == "outlier_tolerance_m") tracking_settings_.outlier_tolerance_m = std::stof(value);
        }
        else if (section == "Window") {
            // NEW naming convention: _p for pixels
            if (key == "width_p") window_settings_.width = std::stoi(value);
            else if (key == "height_p") window_settings_.height = std::stoi(value);
            // BACKWARD COMPATIBILITY: old parameter names without suffix
            else if (key == "width") window_settings_.width = std::stoi(value);
            else if (key == "height") window_settings_.height = std::stoi(value);
        }
        else if (section == "Simulation") {
            // NEW naming convention: _p for pixels
            if (key == "window_width_p") simulation_settings_.window_width = std::stoi(value);
            else if (key == "window_height_p") simulation_settings_.window_height = std::stoi(value);
            // BACKWARD COMPATIBILITY: old parameter names without suffix
            else if (key == "window_width") simulation_settings_.window_width = std::stoi(value);
            else if (key == "window_height") simulation_settings_.window_height = std::stoi(value);
            // Other simulation parameters
            else if (key == "fullscreen") simulation_settings_.fullscreen = (std::stoi(value) != 0);
            else if (key == "num_objects") simulation_settings_.num_objects = std::stoi(value);
            else if (key == "sphere_diameter_m") simulation_settings_.sphere_diameter_m = std::stof(value);
            // LED cluster configuration (NEW)
            else if (key == "use_led_clusters") simulation_settings_.use_led_clusters = (std::stoi(value) != 0);
            else if (key == "cluster_count") simulation_settings_.cluster_count = std::stoi(value);
            else if (key == "leds_per_cluster") simulation_settings_.leds_per_cluster = std::stoi(value);
            else if (key == "cluster_separation_m") simulation_settings_.cluster_separation_m = std::stof(value);
            // Legacy LED configuration
            else if (key == "led_count") simulation_settings_.led_count = std::stoi(value);
            else if (key == "led_radius_m") simulation_settings_.led_radius_m = std::stof(value);
            else if (key == "distance_from_screen_m") simulation_settings_.distance_from_screen_m = std::stof(value);
        }
        else if (section == "ClusterTracking") {
            if (key == "min_leds_per_cluster") cluster_tracking_settings_.min_leds_per_cluster = std::stoi(value);
            else if (key == "min_clusters_for_tracking") cluster_tracking_settings_.min_clusters_for_tracking = std::stoi(value);
            else if (key == "max_cluster_match_distance_m") cluster_tracking_settings_.max_cluster_match_distance_m = std::stof(value);
            else if (key == "use_color_validation") cluster_tracking_settings_.use_color_validation = (std::stoi(value) != 0);
            else if (key == "color_tolerance") cluster_tracking_settings_.color_tolerance = std::stof(value);
        }
        else if (section == "Debug") {
            if (key == "verbosity") debug_settings_.verbosity = std::stoi(value);
            else if (key == "show_projection_matrices") debug_settings_.show_projection_matrices = (std::stoi(value) != 0);
            else if (key == "show_triangulation_details") debug_settings_.show_triangulation_details = (std::stoi(value) != 0);
            else if (key == "show_pose_estimation") debug_settings_.show_pose_estimation = (std::stoi(value) != 0);
            else if (key == "show_coordinate_transform") debug_settings_.show_coordinate_transform = (std::stoi(value) != 0);
        }
        else if (section == "Rendering.3DView") {
            if (key == "background_color_rgb") rendering_3d_view_.background_color_rgb = parse_rgb(value);
            else if (key == "grid_color_rgb") rendering_3d_view_.grid_color_rgb = parse_rgb(value);
            else if (key == "grid_size_m") rendering_3d_view_.grid_size_m = std::stof(value);
            else if (key == "grid_step_m") rendering_3d_view_.grid_step_m = std::stof(value);
            else if (key == "axes_font_scale") rendering_3d_view_.axes_font_scale = std::stof(value);
            else if (key == "axes_line_thickness") rendering_3d_view_.axes_line_thickness = std::stoi(value);
            else if (key == "axes_length_multiplier") rendering_3d_view_.axes_length_multiplier = std::stof(value);
            else if (key == "axes_arrow_thickness") rendering_3d_view_.axes_arrow_thickness = std::stoi(value);
            else if (key == "axes_arrow_tip_ratio") rendering_3d_view_.axes_arrow_tip_ratio = std::stof(value);
        }
        else if (section == "Rendering.Camera") {
            if (key == "left_frustum_color_rgb") rendering_camera_.left_frustum_color_rgb = parse_rgb(value);
            else if (key == "right_frustum_color_rgb") rendering_camera_.right_frustum_color_rgb = parse_rgb(value);
            else if (key == "left_fov_color_rgb") rendering_camera_.left_fov_color_rgb = parse_rgb(value);
            else if (key == "right_fov_color_rgb") rendering_camera_.right_fov_color_rgb = parse_rgb(value);
            else if (key == "frustum_size_m") rendering_camera_.frustum_size_m = std::stof(value);
            else if (key == "frustum_size_small_m") rendering_camera_.frustum_size_small_m = std::stof(value);
            else if (key == "frustum_line_thickness") rendering_camera_.frustum_line_thickness = std::stoi(value);
            else if (key == "fov_line_thickness") rendering_camera_.fov_line_thickness = std::stoi(value);
            else if (key == "fov_plane_alpha") rendering_camera_.fov_plane_alpha = std::stof(value);
            else if (key == "fov_plane_near_dist_m") rendering_camera_.fov_plane_near_dist_m = std::stof(value);
            else if (key == "fov_label_font_scale") rendering_camera_.fov_label_font_scale = std::stof(value);
            else if (key == "fov_label_thickness") rendering_camera_.fov_label_thickness = std::stoi(value);
            else if (key == "fov_label_distance_ratio") rendering_camera_.fov_label_distance_ratio = std::stof(value);
        }
        else if (section == "Rendering.GroundTruth") {
            if (key == "sphere_fill_color_rgb") rendering_ground_truth_.sphere_fill_color_rgb = parse_rgb(value);
            else if (key == "sphere_outline_color_rgb") rendering_ground_truth_.sphere_outline_color_rgb = parse_rgb(value);
            else if (key == "sphere_outline_thickness") rendering_ground_truth_.sphere_outline_thickness = std::stoi(value);
            else if (key == "led_radius_p") rendering_ground_truth_.led_radius_p = std::stoi(value);
            else if (key == "led_outline_radius_p") rendering_ground_truth_.led_outline_radius_p = std::stoi(value);
            else if (key == "led_fill_color_rgb") rendering_ground_truth_.led_fill_color_rgb = parse_rgb(value);
            else if (key == "led_outline_color_rgb") rendering_ground_truth_.led_outline_color_rgb = parse_rgb(value);
            else if (key == "led_outline_thickness") rendering_ground_truth_.led_outline_thickness = std::stoi(value);
        }
        else if (section == "Rendering.Estimated") {
            if (key == "sphere_outline_color_rgb") rendering_estimated_.sphere_outline_color_rgb = parse_rgb(value);
            else if (key == "sphere_outline_thickness") rendering_estimated_.sphere_outline_thickness = std::stoi(value);
            else if (key == "sphere_inner_ratio") rendering_estimated_.sphere_inner_ratio = std::stof(value);
            else if (key == "sphere_inner_color_rgb") rendering_estimated_.sphere_inner_color_rgb = parse_rgb(value);
            else if (key == "sphere_inner_thickness") rendering_estimated_.sphere_inner_thickness = std::stoi(value);
            else if (key == "min_confidence") rendering_estimated_.min_confidence = std::stof(value);
        }
        else if (section == "Rendering.Text") {
            if (key == "label_font_scale") rendering_text_.label_font_scale = std::stof(value);
            else if (key == "label_thickness") rendering_text_.label_thickness = std::stoi(value);
            else if (key == "label_background_color_rgb") rendering_text_.label_background_color_rgb = parse_rgb(value);
            else if (key == "tracking_info_font_scale") rendering_text_.tracking_info_font_scale = std::stof(value);
            else if (key == "tracking_info_thickness") rendering_text_.tracking_info_thickness = std::stoi(value);
            else if (key == "tracking_info_color_rgb") rendering_text_.tracking_info_color_rgb = parse_rgb(value);
            else if (key == "error_excellent_color_rgb") rendering_text_.error_excellent_color_rgb = parse_rgb(value);
            else if (key == "tracking_lost_color_rgb") rendering_text_.tracking_lost_color_rgb = parse_rgb(value);
        }
        else if (section == "Algorithm") {
            if (key == "periodic_debug_interval") algorithm_settings_.periodic_debug_interval = std::stoi(value);
            else if (key == "triangulation_debug_interval") algorithm_settings_.triangulation_debug_interval = std::stoi(value);
            else if (key == "pose_debug_interval") algorithm_settings_.pose_debug_interval = std::stoi(value);
            else if (key == "debug_match_limit") algorithm_settings_.debug_match_limit = std::stoi(value);
        }
        else if (section == "GeneticAlgorithm") {
            if (key == "population_size") ga_settings_.population_size = std::stoi(value);
            else if (key == "num_generations") ga_settings_.num_generations = std::stoi(value);
            else if (key == "mutation_rate") ga_settings_.mutation_rate = std::stof(value);
            else if (key == "crossover_rate") ga_settings_.crossover_rate = std::stof(value);
            else if (key == "frames_per_eval") ga_settings_.frames_per_eval = std::stoi(value);
            else if (key == "optimize_bias_diff") ga_settings_.optimize_bias_diff = (std::stoi(value) != 0);
            else if (key == "optimize_bias_refr") ga_settings_.optimize_bias_refr = (std::stoi(value) != 0);
            else if (key == "optimize_bias_fo") ga_settings_.optimize_bias_fo = (std::stoi(value) != 0);
            else if (key == "optimize_bias_hpf") ga_settings_.optimize_bias_hpf = (std::stoi(value) != 0);
            else if (key == "optimize_accumulation") ga_settings_.optimize_accumulation = (std::stoi(value) != 0);
            else if (key == "optimize_trail_filter") ga_settings_.optimize_trail_filter = (std::stoi(value) != 0);
            else if (key == "optimize_antiflicker") ga_settings_.optimize_antiflicker = (std::stoi(value) != 0);
            else if (key == "optimize_erc") ga_settings_.optimize_erc = (std::stoi(value) != 0);
            else if (key == "enable_cluster_filter") ga_settings_.enable_cluster_filter = (std::stoi(value) != 0);
            else if (key == "cluster_radius") ga_settings_.cluster_radius = std::stoi(value);
            else if (key == "min_cluster_radius") ga_settings_.min_cluster_radius = std::stoi(value);
            else if (key == "cluster_centers") {
                // Parse cluster centers format: "x1,y1;x2,y2;x3,y3"
                ga_settings_.cluster_centers.clear();
                std::stringstream ss(value);
                std::string pair_str;
                while (std::getline(ss, pair_str, ';')) {
                    std::stringstream pair_ss(pair_str);
                    std::string x_str, y_str;
                    if (std::getline(pair_ss, x_str, ',') && std::getline(pair_ss, y_str, ',')) {
                        try {
                            int x = std::stoi(x_str);
                            int y = std::stoi(y_str);
                            ga_settings_.cluster_centers.emplace_back(x, y);
                        } catch (...) {
                            // Skip invalid pairs
                        }
                    }
                }
            }
        }
        else if (section == "Runtime") {
            if (key == "max_event_age_us") runtime_settings_.max_event_age_us = std::stoll(value);
            else if (key == "ga_frame_capture_wait_ms") runtime_settings_.ga_frame_capture_wait_ms = std::stoi(value);
            else if (key == "ga_frame_capture_max_attempts") runtime_settings_.ga_frame_capture_max_attempts = std::stoi(value);
            else if (key == "ga_parameter_settle_ms") runtime_settings_.ga_parameter_settle_ms = std::stoi(value);
            else if (key == "simulation_frame_delay_ms") runtime_settings_.simulation_frame_delay_ms = std::stoi(value);
        }
    }

    std::cout << "Configuration loaded from: " << filename << std::endl;
    std::cout << "  Camera bias settings loaded" << std::endl;
    std::cout << "  Frame accumulation time: " << camera_settings_.accumulation_time_s << "s" << std::endl;
    std::cout << "  GA population size: " << ga_settings_.population_size << std::endl;
    std::cout << "  GA generations: " << ga_settings_.num_generations << std::endl;

    return true;
}
