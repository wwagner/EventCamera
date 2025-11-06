# EventCamera Refactoring Plan

**Goal**: Make the codebase modular and extensible so we can easily experiment with new image processing techniques, camera features, and UI components.

**Current State**:
- main.cpp: 1,938 lines, 90KB
- Most functionality in a single 1,333-line main() function
- 19+ global state variables
- Repeated patterns throughout

**Target State**:
- Modular architecture with clear separation of concerns
- Easy to add new image processing filters
- Easy to add new camera features/controls
- Easy to test individual components
- main.cpp reduced to ~400-500 lines (orchestration only)

---

## Architecture Vision

```
EventCamera Application
├── Core/
│   ├── AppState           - Centralized application state
│   ├── CameraManager      - Camera lifecycle (already exists, enhance)
│   └── AppConfig          - Configuration (already exists)
│
├── Video/
│   ├── FrameProcessor     - Image processing pipeline
│   ├── TextureManager     - OpenGL texture management
│   ├── FrameBuffer        - Thread-safe frame storage
│   └── VideoFilters/      - Pluggable filter system
│       ├── IVideoFilter   - Interface for all filters
│       ├── ROIFilter      - Region of interest cropping
│       ├── SubtractionFilter - Frame differencing
│       └── [Future filters...]
│
├── Camera/
│   ├── CameraController   - Bias settings, parameter application
│   ├── HardwareFeatures/  - Modular feature system
│   │   ├── IHardwareFeature - Interface for camera features
│   │   ├── ROIFeature     - Region of Interest
│   │   ├── ERCFeature     - Event Rate Controller
│   │   ├── AntiFlickerFeature
│   │   ├── TrailFilterFeature
│   │   ├── DigitalCropFeature
│   │   └── MonitoringFeature
│   └── BiasManager        - Centralized bias control
│
├── UI/
│   ├── UIPanel (base)     - Common panel functionality
│   ├── SettingsPanel      - Camera settings UI
│   ├── CameraFeedPanel    - Live feed display
│   ├── FeaturesPanel      - Hardware features UI
│   └── GAPanel            - Genetic algorithm UI
│
└── Optimization/
    ├── EventCameraGeneticOptimizer (already exists)
    └── GAController       - GA integration and lifecycle
```

---

## Phase 1: Extract Video Processing Pipeline

**Goal**: Separate all frame processing logic from main.cpp to enable easy addition of new image processing effects.

**Duration**: 2-3 hours
**Risk**: Low (well-defined interfaces)
**Value**: High (enables experimentation with filters)

### 1.1 Create Video Module Structure

**New Files**:
```
include/video/
  ├── frame_buffer.h
  ├── frame_processor.h
  ├── texture_manager.h
  └── filters/
      ├── video_filter.h      (interface)
      ├── roi_filter.h
      └── subtraction_filter.h

src/video/
  ├── frame_buffer.cpp
  ├── frame_processor.cpp
  ├── texture_manager.cpp
  └── filters/
      ├── roi_filter.cpp
      └── subtraction_filter.cpp
```

### 1.2 FrameBuffer Class

**Purpose**: Thread-safe frame storage with frame dropping

**Interface**:
```cpp
class FrameBuffer {
public:
    // Store new frame (may drop if not consumed)
    void store_frame(const cv::Mat& frame);

    // Consume frame for display
    std::optional<cv::Mat> consume_frame();

    // Check if frame is ready
    bool has_unconsumed_frame() const;

    // Statistics
    int64_t get_frames_dropped() const;
    int64_t get_frames_generated() const;

private:
    cv::Mat current_frame_;
    std::atomic<bool> frame_consumed_{true};
    std::atomic<int64_t> frames_dropped_{0};
    std::atomic<int64_t> frames_generated_{0};
    std::mutex mutex_;
};
```

**Replaces**: Lines 44, 49-50, 56-58, 107-119 in main.cpp

### 1.3 IVideoFilter Interface

**Purpose**: Pluggable filter system for image processing

**Interface**:
```cpp
class IVideoFilter {
public:
    virtual ~IVideoFilter() = default;

    // Apply filter to frame
    virtual cv::Mat apply(const cv::Mat& input) = 0;

    // Enable/disable filter
    virtual void set_enabled(bool enabled) = 0;
    virtual bool is_enabled() const = 0;

    // Filter metadata
    virtual std::string name() const = 0;
    virtual std::string description() const = 0;
};
```

### 1.4 Concrete Filters

**ROIFilter**:
```cpp
class ROIFilter : public IVideoFilter {
public:
    void set_roi(int x, int y, int width, int height);
    void set_crop_to_roi(bool crop);
    void set_show_rectangle(bool show);

    cv::Mat apply(const cv::Mat& input) override;

private:
    bool enabled_ = false;
    bool crop_to_roi_ = false;
    bool show_rectangle_ = false;
    int x_ = 0, y_ = 0, width_ = 640, height_ = 360;
    std::mutex mutex_;
};
```

**SubtractionFilter**:
```cpp
class SubtractionFilter : public IVideoFilter {
public:
    cv::Mat apply(const cv::Mat& input) override;
    void reset(); // Clear previous frame

private:
    bool enabled_ = false;
    cv::Mat previous_frame_;
    std::mutex mutex_;
};
```

**Replaces**: Lines 63-76, 126-206 in main.cpp

### 1.5 FrameProcessor Class

**Purpose**: Orchestrate filter pipeline and frame transformations

**Interface**:
```cpp
class FrameProcessor {
public:
    // Add filter to pipeline
    void add_filter(std::shared_ptr<IVideoFilter> filter);
    void remove_filter(const std::string& name);

    // Get filter by name
    std::shared_ptr<IVideoFilter> get_filter(const std::string& name);

    // Process frame through pipeline
    cv::Mat process(const cv::Mat& input);

    // Utility conversions
    static cv::Mat bgr_to_rgb(const cv::Mat& frame);

private:
    std::vector<std::shared_ptr<IVideoFilter>> filters_;
    std::mutex mutex_;
};
```

### 1.6 TextureManager Class

**Purpose**: OpenGL texture handling

**Interface**:
```cpp
class TextureManager {
public:
    TextureManager();
    ~TextureManager();

    // Upload frame to GPU
    void upload_frame(const cv::Mat& rgb_frame);

    // Get texture ID for rendering
    GLuint get_texture_id() const { return texture_id_; }

    // Get current dimensions
    int get_width() const { return width_; }
    int get_height() const { return height_; }

private:
    GLuint texture_id_ = 0;
    int width_ = 0;
    int height_ = 0;
};
```

**Replaces**: Lines 59, 168-234 in main.cpp

### 1.7 Benefits After Phase 1

✅ **Easy to add new filters** - Just implement IVideoFilter
✅ **Testable** - Can unit test filters independently
✅ **Reusable** - Filter system can be used in other projects
✅ **Clear ownership** - No more scattered global state for video

**Example: Adding a new blur filter**:
```cpp
class BlurFilter : public IVideoFilter {
    cv::Mat apply(const cv::Mat& input) override {
        if (!enabled_) return input;
        cv::Mat blurred;
        cv::GaussianBlur(input, blurred, cv::Size(kernel_size_, kernel_size_), 0);
        return blurred;
    }
};

// In main:
auto blur = std::make_shared<BlurFilter>();
frame_processor.add_filter(blur);
```

---

## Phase 2: Consolidate Application State

**Goal**: Eliminate global state and create a centralized, thread-safe state manager.

**Duration**: 2-3 hours
**Risk**: Medium (touches many areas)
**Value**: Very High (eliminates bugs, improves maintainability)

### 2.1 Create Core Module Structure

**New Files**:
```
include/core/
  ├── app_state.h
  ├── frame_sync.h
  ├── event_metrics.h
  └── display_settings.h

src/core/
  ├── app_state.cpp
  ├── frame_sync.cpp
  ├── event_metrics.cpp
  └── display_settings.cpp
```

### 2.2 FrameSync Class

**Purpose**: Frame timing and synchronization

**Interface**:
```cpp
class FrameSync {
public:
    // Update timestamps
    void on_frame_generated(int64_t camera_ts, int64_t system_ts);
    void on_frame_displayed(int64_t system_ts);

    // Query state
    int64_t get_last_frame_camera_ts() const;
    int64_t get_last_frame_system_ts() const;
    int64_t get_last_display_time_us() const;

    // Frame rate limiting
    bool should_display_frame() const;

private:
    std::atomic<int64_t> last_frame_camera_ts_{0};
    std::atomic<int64_t> last_frame_system_ts_{0};
    std::atomic<int64_t> last_display_time_us_{0};
};
```

**Replaces**: Lines 45-47 in main.cpp

### 2.3 EventMetrics Class

**Purpose**: Event rate tracking and statistics

**Interface**:
```cpp
class EventMetrics {
public:
    // Record events
    void record_events(int64_t count, int64_t timestamp_us);
    void record_event_timestamp(int64_t timestamp_us);

    // Query metrics
    int64_t get_total_events() const;
    int64_t get_events_per_second() const;
    int64_t get_last_event_timestamp() const;

    // Reset
    void reset();

private:
    std::atomic<int64_t> total_events_received_{0};
    std::atomic<int64_t> events_last_second_{0};
    std::atomic<int64_t> last_event_rate_update_us_{0};
    std::atomic<int64_t> last_event_camera_ts_{0};
    std::mutex mutex_;
};
```

**Replaces**: Lines 48, 53-55 in main.cpp

### 2.4 DisplaySettings Class

**Purpose**: Display configuration

**Interface**:
```cpp
class DisplaySettings {
public:
    // Display rate
    void set_target_fps(int fps);
    int get_target_fps() const;

    // Window size
    void set_window_size(int width, int height);
    int get_window_width() const;
    int get_window_height() const;

private:
    std::atomic<int> target_display_fps_{10};
    int window_width_ = 1280;
    int window_height_ = 720;
};
```

**Replaces**: Lines 51, 60-61 in main.cpp

### 2.5 CameraState Class

**Purpose**: Camera connection state

**Interface**:
```cpp
class CameraState {
public:
    // Connection management
    void set_connected(bool connected);
    bool is_connected() const;

    void set_simulation_mode(bool enabled);
    bool is_simulation_mode() const;

    // Camera start time
    void set_camera_start_time_us(int64_t time_us);
    int64_t get_camera_start_time_us() const;

    // Camera objects (managed pointers)
    std::unique_ptr<CameraManager>& camera_manager();
    std::unique_ptr<Metavision::PeriodicFrameGenerationAlgorithm>& frame_generator();

private:
    std::unique_ptr<CameraManager> camera_mgr_;
    std::unique_ptr<Metavision::PeriodicFrameGenerationAlgorithm> frame_gen_;
    std::atomic<bool> camera_connected_{false};
    std::atomic<bool> simulation_mode_{false};
    std::atomic<int64_t> camera_start_time_us_{0};
    std::mutex connection_mutex_;
};
```

**Replaces**: Lines 52, 93-101 in main.cpp

### 2.6 AppState Class (Top-Level)

**Purpose**: Unified state container

**Interface**:
```cpp
class AppState {
public:
    AppState();
    ~AppState();

    // Subsystems
    FrameBuffer& frame_buffer();
    FrameProcessor& frame_processor();
    TextureManager& texture_manager();
    FrameSync& frame_sync();
    EventMetrics& event_metrics();
    DisplaySettings& display_settings();
    CameraState& camera_state();

    // Running state
    bool is_running() const;
    void request_shutdown();

private:
    std::atomic<bool> running_{true};

    // Subsystem instances
    std::unique_ptr<FrameBuffer> frame_buffer_;
    std::unique_ptr<FrameProcessor> frame_processor_;
    std::unique_ptr<TextureManager> texture_manager_;
    std::unique_ptr<FrameSync> frame_sync_;
    std::unique_ptr<EventMetrics> event_metrics_;
    std::unique_ptr<DisplaySettings> display_settings_;
    std::unique_ptr<CameraState> camera_state_;
};
```

### 2.7 Benefits After Phase 2

✅ **No more globals** - All state in one place
✅ **Thread-safe by design** - Proper synchronization
✅ **Easy to reason about** - Clear ownership and lifecycle
✅ **Easy to extend** - Add new subsystems as needed
✅ **Testable** - Can create AppState for testing

---

## Phase 3: Modular Hardware Features System

**Goal**: Replace 400+ lines of repetitive UI code with a pluggable feature system.

**Duration**: 4-5 hours
**Risk**: Medium (significant restructuring)
**Value**: Very High (biggest code reduction, easiest to extend)

### 3.1 Create Camera Module Structure

**New Files**:
```
include/camera/
  ├── hardware_feature.h      (interface)
  ├── feature_manager.h
  ├── bias_manager.h
  ├── camera_controller.h
  └── features/
      ├── roi_feature.h
      ├── erc_feature.h
      ├── antiflicker_feature.h
      ├── trail_filter_feature.h
      ├── digital_crop_feature.h
      └── monitoring_feature.h

src/camera/
  ├── feature_manager.cpp
  ├── bias_manager.cpp
  ├── camera_controller.cpp
  └── features/
      ├── roi_feature.cpp
      ├── erc_feature.cpp
      ├── antiflicker_feature.cpp
      ├── trail_filter_feature.cpp
      ├── digital_crop_feature.cpp
      └── monitoring_feature.cpp
```

### 3.2 IHardwareFeature Interface

**Purpose**: Unified interface for all camera hardware features

**Interface**:
```cpp
class IHardwareFeature {
public:
    virtual ~IHardwareFeature() = default;

    // Lifecycle
    virtual bool initialize(Metavision::Camera& camera) = 0;
    virtual void shutdown() = 0;

    // Availability
    virtual bool is_available() const = 0;
    virtual bool is_enabled() const = 0;

    // Control
    virtual void enable(bool enabled) = 0;
    virtual void apply_settings() = 0;

    // Metadata
    virtual std::string name() const = 0;
    virtual std::string description() const = 0;
    virtual FeatureCategory category() const = 0;

    // UI Rendering (returns true if settings changed)
    virtual bool render_ui() = 0;
};

enum class FeatureCategory {
    Monitoring,      // Read-only hardware info
    RegionControl,   // ROI, Digital Crop
    EventFiltering,  // ERC, Anti-Flicker, Trail Filter
    Advanced         // Other features
};
```

### 3.3 Example Feature Implementation: ERC

**ERCFeature**:
```cpp
class ERCFeature : public IHardwareFeature {
public:
    bool initialize(Metavision::Camera& camera) override {
        erc_ = camera.get_device().get_facility<Metavision::I_ERC_Module>();
        return erc_ != nullptr;
    }

    bool render_ui() override {
        if (!is_available()) return false;

        bool changed = false;

        if (ImGui::CollapsingHeader("Event Rate Controller (ERC)")) {
            ImGui::TextWrapped(description().c_str());
            ImGui::Spacing();

            bool enabled = is_enabled();
            if (ImGui::Checkbox("Enable ERC", &enabled)) {
                enable(enabled);
                changed = true;
            }

            if (enabled) {
                int rate_kevps = target_rate_kevps_;
                if (ImGui::SliderInt("Event Rate (k events/s)", &rate_kevps, 1, 200)) {
                    target_rate_kevps_ = rate_kevps;
                    changed = true;
                }

                if (changed) {
                    apply_settings();
                }
            }
        }

        return changed;
    }

    void apply_settings() override {
        if (!is_available() || !is_enabled()) return;

        try {
            erc_->set_cd_event_rate(target_rate_kevps_ * 1000);
            std::cout << "ERC rate set to " << target_rate_kevps_ << " kevps" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to set ERC rate: " << e.what() << std::endl;
        }
    }

    std::string name() const override { return "ERC"; }
    std::string description() const override {
        return "Limit maximum event rate to prevent bandwidth saturation.";
    }
    FeatureCategory category() const override { return FeatureCategory::EventFiltering; }

private:
    Metavision::I_ERC_Module* erc_ = nullptr;
    bool enabled_ = false;
    int target_rate_kevps_ = 100;
};
```

**Replaces**: Lines 1333-1367 in main.cpp (~35 lines → ~50 lines with more functionality)

### 3.4 FeatureManager Class

**Purpose**: Manage collection of hardware features

**Interface**:
```cpp
class FeatureManager {
public:
    // Add feature to manager
    void register_feature(std::shared_ptr<IHardwareFeature> feature);

    // Initialize all features with camera
    void initialize_all(Metavision::Camera& camera);
    void shutdown_all();

    // Get features by category
    std::vector<std::shared_ptr<IHardwareFeature>>
        get_features_by_category(FeatureCategory category);

    // Get feature by name
    std::shared_ptr<IHardwareFeature> get_feature(const std::string& name);

    // Render all feature UIs
    void render_all_ui();

    // Apply all settings
    void apply_all_settings();

private:
    std::vector<std::shared_ptr<IHardwareFeature>> features_;
    std::mutex mutex_;
};
```

### 3.5 BiasManager Class

**Purpose**: Centralized bias control (replaces scattered bias code)

**Interface**:
```cpp
class BiasManager {
public:
    struct BiasRange {
        int min;
        int max;
        int current;
    };

    // Initialize with camera
    bool initialize(Metavision::Camera& camera);

    // Get available biases and their ranges
    std::map<std::string, BiasRange> get_bias_ranges() const;

    // Set bias value (with validation)
    bool set_bias(const std::string& name, int value);
    int get_bias(const std::string& name) const;

    // Apply biases to camera
    void apply_to_camera();

    // Reset to defaults (middle of range)
    void reset_to_defaults();

    // UI helpers
    bool render_bias_ui(const std::string& name, const std::string& label);

private:
    Metavision::I_LL_Biases* ll_biases_ = nullptr;
    std::map<std::string, BiasRange> bias_ranges_;
    std::mutex mutex_;

    // Exponential mapping for UI
    int slider_to_bias(float slider_pct, int min, int max) const;
    float bias_to_slider(int bias, int min, int max) const;
};
```

**Replaces**: Lines 239-274, 658-751, 1036-1112 in main.cpp

### 3.6 Feature Registration (in main)

**Usage**:
```cpp
// Create feature manager
auto feature_mgr = std::make_unique<FeatureManager>();

// Register all features
feature_mgr->register_feature(std::make_shared<MonitoringFeature>());
feature_mgr->register_feature(std::make_shared<ROIFeature>());
feature_mgr->register_feature(std::make_shared<ERCFeature>());
feature_mgr->register_feature(std::make_shared<AntiFlickerFeature>());
feature_mgr->register_feature(std::make_shared<TrailFilterFeature>());
feature_mgr->register_feature(std::make_shared<DigitalCropFeature>());

// Initialize with camera
feature_mgr->initialize_all(*camera);

// In UI render loop
feature_mgr->render_all_ui();
```

### 3.7 Benefits After Phase 3

✅ **~400 lines → ~50 lines** in main UI loop
✅ **Easy to add features** - Just implement interface and register
✅ **No more copy-paste** - Feature boilerplate eliminated
✅ **Self-documenting** - Each feature is self-contained
✅ **Testable** - Can test features independently

**Example: Adding a new trigger feature**:
```cpp
class TriggerFeature : public IHardwareFeature {
    // Implement interface methods
};

// In main:
feature_mgr->register_feature(std::make_shared<TriggerFeature>());
// Done! UI and functionality automatically integrated
```

---

## Phase 4: UI Panel System

**Goal**: Separate UI rendering into modular panels.

**Duration**: 3-4 hours
**Risk**: Low (mostly moving code)
**Value**: High (cleaner main loop, reusable panels)

### 4.1 Create UI Module Structure

**New Files**:
```
include/ui/
  ├── ui_panel.h          (base class)
  ├── settings_panel.h
  ├── camera_feed_panel.h
  ├── features_panel.h
  └── ga_panel.h

src/ui/
  ├── ui_panel.cpp
  ├── settings_panel.cpp
  ├── camera_feed_panel.cpp
  ├── features_panel.cpp
  └── ga_panel.cpp
```

### 4.2 UIPanel Base Class

**Purpose**: Common panel functionality

**Interface**:
```cpp
class UIPanel {
public:
    virtual ~UIPanel() = default;

    // Render the panel (called each frame)
    virtual void render() = 0;

    // Panel metadata
    virtual std::string title() const = 0;

    // Visibility
    void set_visible(bool visible) { visible_ = visible; }
    bool is_visible() const { return visible_; }

    // Position and size
    void set_position(ImVec2 pos) { position_ = pos; }
    void set_size(ImVec2 size) { size_ = size; }

protected:
    bool visible_ = true;
    ImVec2 position_{10, 10};
    ImVec2 size_{400, 600};
};
```

### 4.3 SettingsPanel Class

**Purpose**: Camera settings and biases

**Interface**:
```cpp
class SettingsPanel : public UIPanel {
public:
    SettingsPanel(AppState& state,
                  AppConfig& config,
                  BiasManager& bias_mgr);

    void render() override;
    std::string title() const override { return "Camera Settings"; }

private:
    void render_bias_controls();
    void render_display_settings();
    void render_frame_generation();
    void render_apply_button();

    AppState& state_;
    AppConfig& config_;
    BiasManager& bias_mgr_;
};
```

**Replaces**: Lines 916-1180 in main.cpp

### 4.4 FeaturesPanel Class

**Purpose**: Hardware features UI

**Interface**:
```cpp
class FeaturesPanel : public UIPanel {
public:
    FeaturesPanel(FeatureManager& feature_mgr);

    void render() override;
    std::string title() const override { return "Advanced Features"; }

private:
    void render_category(FeatureCategory category);

    FeatureManager& feature_mgr_;
};
```

**Replaces**: Lines 1204-1625 in main.cpp

### 4.5 CameraFeedPanel Class

**Purpose**: Live feed display

**Interface**:
```cpp
class CameraFeedPanel : public UIPanel {
public:
    CameraFeedPanel(AppState& state);

    void render() override;
    std::string title() const override { return "Camera Feed"; }

private:
    void render_statistics();
    void render_feed_texture();

    AppState& state_;
};
```

**Replaces**: Lines 1859-1891 in main.cpp

### 4.6 GAPanel Class

**Purpose**: Genetic algorithm controls

**Interface**:
```cpp
class GAPanel : public UIPanel {
public:
    GAPanel(AppState& state,
            AppConfig& config,
            GAController& ga_ctrl);

    void render() override;
    std::string title() const override { return "Genetic Algorithm"; }

private:
    void render_config();
    void render_controls();
    void render_progress();
    void render_results();

    AppState& state_;
    AppConfig& config_;
    GAController& ga_ctrl_;
};
```

**Replaces**: Lines 1628-1856 in main.cpp

### 4.7 Simplified Main Loop

**After Phase 4**:
```cpp
int main() {
    // Initialize everything
    AppState state;
    AppConfig config;
    // ... other initialization ...

    // Create UI panels
    SettingsPanel settings_panel(state, config, bias_mgr);
    CameraFeedPanel feed_panel(state);
    FeaturesPanel features_panel(feature_mgr);
    GAPanel ga_panel(state, config, ga_ctrl);

    // Main loop
    while (state.is_running() && !glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Render all panels
        settings_panel.render();
        feed_panel.render();
        features_panel.render();
        ga_panel.render();

        ImGui::Render();
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    return 0;
}
```

### 4.8 Benefits After Phase 4

✅ **Clean main loop** - ~50 lines instead of 1,000+
✅ **Reusable panels** - Can be used in other apps
✅ **Easy to reorganize** - Just reorder render calls
✅ **Easy to add panels** - Implement UIPanel interface

---

## Phase 5: Camera Control Abstraction (Optional)

**Goal**: Clean up camera initialization and control.

**Duration**: 2-3 hours
**Risk**: Low
**Value**: Medium (cleanup, removes duplication)

### 5.1 CameraController Class

**Purpose**: High-level camera operations

**Interface**:
```cpp
class CameraController {
public:
    CameraController(AppState& state, AppConfig& config);

    // Connection
    bool connect_camera();
    void disconnect_camera();
    bool is_connected() const;

    // Event processing
    void start_event_loop();
    void stop_event_loop();

    // Settings
    void apply_all_settings();

    // Access
    Metavision::Camera* get_camera();
    BiasManager& bias_manager();
    FeatureManager& feature_manager();

private:
    void setup_event_callbacks();
    void create_frame_generator();

    AppState& state_;
    AppConfig& config_;
    std::unique_ptr<BiasManager> bias_mgr_;
    std::unique_ptr<FeatureManager> feature_mgr_;
    std::unique_ptr<std::thread> event_thread_;
};
```

**Replaces**: Lines 427-599, 605-774, setup code in main.cpp

### 5.2 EventProcessor Class

**Purpose**: Event loop logic

**Interface**:
```cpp
class EventProcessor {
public:
    EventProcessor(AppState& state);

    // Process batch of events
    void process_events(const Metavision::EventCD* begin,
                       const Metavision::EventCD* end);

    // Check if events are too old
    bool is_batch_too_old(int64_t event_ts) const;

    // Update metrics
    void update_event_metrics(const Metavision::EventCD* begin,
                             const Metavision::EventCD* end);

private:
    AppState& state_;
};
```

**Replaces**: Event callback logic in lines 532-594, 825-842

### 5.3 Benefits After Phase 5

✅ **No duplication** - Event thread logic unified
✅ **Clear lifecycle** - Camera initialization in one place
✅ **Easy to test** - Can mock camera connection

---

## Phase 6: Optimization Integration (Optional)

**Goal**: Clean up GA integration.

**Duration**: 1-2 hours
**Risk**: Low
**Value**: Medium (removes duplication)

### 6.1 GAController Class

**Purpose**: Manage GA lifecycle and integration

**Interface**:
```cpp
class GAController {
public:
    GAController(AppState& state,
                 AppConfig& config,
                 CameraController& camera_ctrl);

    // Control
    void start_optimization();
    void stop_optimization();
    bool is_running() const;

    // Results
    float get_best_fitness() const;
    const EventCameraGeneticOptimizer::Genome& get_best_genome() const;
    void apply_best_parameters();

    // Progress
    int get_current_generation() const;
    int get_total_generations() const;

private:
    EventCameraGeneticOptimizer::Genome clamp_genome_to_hardware(
        const EventCameraGeneticOptimizer::Genome& genome);

    AppState& state_;
    AppConfig& config_;
    CameraController& camera_ctrl_;
    std::unique_ptr<EventCameraGeneticOptimizer> optimizer_;
    std::unique_ptr<std::thread> optimizer_thread_;
};
```

**Replaces**: GA state (lines 78-92), duplication in lines 1705-1836

### 6.2 Benefits After Phase 6

✅ **No duplication** - Genome clamping in one place
✅ **Clear interface** - GA control separated from UI

---

## Implementation Strategy

### Order of Implementation

1. **Phase 1** (Video Processing) - Foundation, easy wins
2. **Phase 2** (App State) - Critical refactoring
3. **Phase 3** (Hardware Features) - Biggest code reduction
4. **Phase 4** (UI Panels) - Clean up main loop
5. **Phase 5** (Camera Control) - Optional cleanup
6. **Phase 6** (GA Integration) - Optional cleanup

### Incremental Approach

**For each phase**:
1. Create new files/classes
2. Implement functionality
3. Update CMakeLists.txt
4. Test compilation
5. Update main.cpp to use new classes
6. Test functionality
7. Remove old code
8. Commit changes

### Testing Strategy

**After each phase**:
- ✅ Application compiles
- ✅ Camera connects
- ✅ Live feed displays
- ✅ UI controls work
- ✅ Settings apply correctly
- ✅ No memory leaks (visual check)

---

## Expected Results

### Code Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| main.cpp lines | 1,938 | ~400-500 | -75% |
| main() lines | 1,333 | ~100 | -92% |
| Global variables | 19+ | 0 | -100% |
| Files | 9 | ~30 | +233% |
| Duplicated code | High | Low | -80% |

### Modularity Wins

✅ **Add new image filter**: Implement IVideoFilter (~30 lines)
✅ **Add new camera feature**: Implement IHardwareFeature (~80 lines)
✅ **Add new UI panel**: Implement UIPanel (~50 lines)
✅ **Experiment with processing**: Just add filter to pipeline
✅ **Test components**: Create unit tests for each class

---

## Future Extensibility Examples

### Adding Edge Detection Filter
```cpp
class EdgeDetectionFilter : public IVideoFilter {
    cv::Mat apply(const cv::Mat& input) override {
        if (!enabled_) return input;
        cv::Mat edges;
        cv::Canny(input, edges, threshold1_, threshold2_);
        return edges;
    }
    // ... rest of implementation
};

// Usage: frame_processor.add_filter(std::make_shared<EdgeDetectionFilter>());
```

### Adding Custom Hardware Feature
```cpp
class CustomTriggerFeature : public IHardwareFeature {
    // Implement interface...
};

// Usage: feature_mgr->register_feature(std::make_shared<CustomTriggerFeature>());
```

### Adding Histogram Panel
```cpp
class HistogramPanel : public UIPanel {
    void render() override {
        ImGui::Begin(title().c_str());
        // Render histogram
        ImGui::End();
    }
};

// Usage: HistogramPanel histogram_panel(state);
//        histogram_panel.render();
```

---

## Risk Mitigation

### Backup Strategy
- Create branch before each phase
- Commit after each successful phase
- Keep main.cpp.bak as rollback point

### Fallback Plan
If any phase causes issues:
1. Revert to previous commit
2. Fix issues in isolated test
3. Retry integration

### Testing Checkpoints
- Camera connection still works
- Video feed displays correctly
- All features still function
- Performance not degraded

---

## Time Estimates

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Video Processing | 2-3 hours | 2-3 hours |
| Phase 2: App State | 2-3 hours | 4-6 hours |
| Phase 3: Hardware Features | 4-5 hours | 8-11 hours |
| Phase 4: UI Panels | 3-4 hours | 11-15 hours |
| Phase 5: Camera Control (opt) | 2-3 hours | 13-18 hours |
| Phase 6: GA Integration (opt) | 1-2 hours | 14-20 hours |

**Core Refactoring (Phases 1-4)**: ~11-15 hours
**Full Refactoring (All phases)**: ~14-20 hours

---

## Success Criteria

✅ main.cpp reduced to <500 lines
✅ All functionality preserved
✅ No performance regression
✅ Can add new filter in <30 lines
✅ Can add new feature in <80 lines
✅ Code is self-documenting
✅ Easy to test components individually
✅ Team can understand structure in <30 minutes

---

## Next Steps

To begin refactoring:

1. **Review this plan** - Ensure phases align with goals
2. **Choose starting phase** - Recommend Phase 1 (quick win)
3. **Create branch** - `git checkout -b refactor/modular-architecture`
4. **Begin implementation** - Start with first phase
5. **Test thoroughly** - After each change
6. **Commit frequently** - Small, atomic commits

Ready to start? Which phase would you like to begin with?
