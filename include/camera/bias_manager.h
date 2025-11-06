#ifndef BIAS_MANAGER_H
#define BIAS_MANAGER_H

#include <string>
#include <map>
#include <mutex>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/sdk/driver/camera.h>

namespace EventCamera {

/**
 * @brief Manages camera bias settings
 *
 * Provides centralized access to camera bias controls with range validation
 * and exponential UI mapping for better low-value control.
 */
class BiasManager {
public:
    /**
     * @brief Range information for a bias parameter
     */
    struct BiasRange {
        int min;      // Minimum value supported by hardware
        int max;      // Maximum value supported by hardware
        int current;  // Current value
    };

    BiasManager() = default;
    ~BiasManager() = default;

    /**
     * @brief Initialize the bias manager with a camera
     * @param camera Reference to the Metavision camera
     * @return true if initialization succeeded
     */
    bool initialize(Metavision::Camera& camera);

    /**
     * @brief Get all available bias ranges
     * @return Map of bias name to range information
     */
    const std::map<std::string, BiasRange>& get_bias_ranges() const {
        return bias_ranges_;
    }

    /**
     * @brief Set a bias value (with validation)
     * @param name Name of the bias (e.g., "bias_diff")
     * @param value Value to set (will be clamped to range)
     * @return true if successful
     */
    bool set_bias(const std::string& name, int value);

    /**
     * @brief Get current value of a bias
     * @param name Name of the bias
     * @return Current value, or 0 if not found
     */
    int get_bias(const std::string& name) const;

    /**
     * @brief Apply all pending bias changes to camera
     */
    void apply_to_camera();

    /**
     * @brief Reset all biases to defaults (middle of range)
     */
    void reset_to_defaults();

    /**
     * @brief Render ImGui UI for a specific bias
     * @param name Name of the bias
     * @param label Display label for the UI
     * @param description Description text for the bias
     * @return true if value changed
     */
    bool render_bias_ui(const std::string& name, const std::string& label,
                       const std::string& description = "");

    /**
     * @brief Check if bias manager is initialized
     */
    bool is_initialized() const { return ll_biases_ != nullptr; }

    /**
     * @brief Set up default bias ranges for simulation mode
     */
    void setup_simulation_defaults();

private:
    /**
     * @brief Convert slider percentage (0-100) to bias value using exponential mapping
     * @param slider_pct Slider position (0-100)
     * @param min Minimum bias value
     * @param max Maximum bias value
     * @return Bias value
     */
    int slider_to_bias(float slider_pct, int min, int max) const;

    /**
     * @brief Convert bias value to slider percentage using inverse exponential mapping
     * @param bias Bias value
     * @param min Minimum bias value
     * @param max Maximum bias value
     * @return Slider position (0-100)
     */
    float bias_to_slider(int bias, int min, int max) const;

    Metavision::I_LL_Biases* ll_biases_ = nullptr;
    std::map<std::string, BiasRange> bias_ranges_;
    std::map<std::string, float> slider_positions_;  // Track slider state for UI
    mutable std::mutex mutex_;
};

} // namespace EventCamera

#endif // BIAS_MANAGER_H
