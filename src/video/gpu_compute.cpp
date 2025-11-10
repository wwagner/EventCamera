#include "video/gpu_compute.h"
#include <iostream>
#include <cstring>

namespace video {
namespace gpu {

//=============================================================================
// Compute Shader Sources
//=============================================================================

// Morphology compute shader (erode/dilate)
const char* morphology_shader_source = R"(
#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, r8) uniform readonly image2D input_image;
layout(binding = 1, r8) uniform writeonly image2D output_image;

uniform int kernel_size;
uniform int operation;  // 0=erode, 1=dilate

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    // Initialize result based on operation
    float result = (operation == 0) ? 1.0 : 0.0;
    int half_kernel = kernel_size / 2;

    // Apply morphology kernel
    for (int y = -half_kernel; y <= half_kernel; y++) {
        for (int x = -half_kernel; x <= half_kernel; x++) {
            ivec2 sample_pos = pos + ivec2(x, y);
            sample_pos = clamp(sample_pos, ivec2(0), size - 1);

            float val = imageLoad(input_image, sample_pos).r;

            if (operation == 0) {
                result = min(result, val);  // Erode: minimum
            } else {
                result = max(result, val);  // Dilate: maximum
            }
        }
    }

    imageStore(output_image, pos, vec4(result));
}
)";

// Histogram compute shader with atomic operations
const char* histogram_shader_source = R"(
#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, r8) uniform readonly image2D input_image;

// SSBO for histogram (256 bins)
layout(std430, binding = 1) buffer HistogramBuffer {
    uint histogram[256];
};

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pos.x >= size.x || pos.y >= size.y) return;

    // Read pixel value
    float val = imageLoad(input_image, pos).r;
    uint bin = uint(val * 255.0);

    // Atomic increment histogram bin
    atomicAdd(histogram[bin], 1u);
}
)";

// Fitness evaluation compute shader
const char* fitness_shader_source = R"(
#version 430 core
#extension GL_NV_shader_atomic_float : enable

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, r8) uniform readonly image2D input_image;

// SSBO for fitness metrics
layout(std430, binding = 1) buffer MetricsBuffer {
    float mean_brightness;
    float variance;
    float non_zero_pixels;
    float total_pixels;
};

// Shared memory for reduction
shared float local_sum[256];
shared float local_sum_sq[256];
shared uint local_count[256];

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);
    uint local_idx = gl_LocalInvocationIndex;

    // Initialize shared memory
    local_sum[local_idx] = 0.0;
    local_sum_sq[local_idx] = 0.0;
    local_count[local_idx] = 0u;
    barrier();

    // Process pixel
    if (pos.x < size.x && pos.y < size.y) {
        float val = imageLoad(input_image, pos).r;
        local_sum[local_idx] = val;
        local_sum_sq[local_idx] = val * val;
        local_count[local_idx] = (val > 0.04) ? 1u : 0u;  // Threshold ~10/255
    }
    barrier();

    // Parallel reduction (only first thread in workgroup)
    if (local_idx == 0) {
        float sum = 0.0;
        float sum_sq = 0.0;
        uint count = 0u;

        for (uint i = 0; i < 256; i++) {
            sum += local_sum[i];
            sum_sq += local_sum_sq[i];
            count += local_count[i];
        }

        // Atomic add to global results
        atomicAdd(mean_brightness, sum);
        atomicAdd(variance, sum_sq);
        atomicAdd(non_zero_pixels, float(count));
    }
}
)";

//=============================================================================
// Utility Functions
//=============================================================================

GLuint compile_compute_shader(const char* source) {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    if (!check_compute_errors(shader, "SHADER")) {
        glDeleteShader(shader);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);

    if (!check_compute_errors(program, "PROGRAM")) {
        glDeleteProgram(program);
        glDeleteShader(shader);
        return 0;
    }

    glDeleteShader(shader);
    return program;
}

bool check_compute_errors(GLuint shader, const char* type) {
    GLint success;
    GLchar info_log[1024];

    if (strcmp(type, "PROGRAM") != 0) {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, nullptr, info_log);
            std::cerr << "ERROR::COMPUTE_SHADER::COMPILATION_FAILED\n"
                     << info_log << std::endl;
            return false;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, nullptr, info_log);
            std::cerr << "ERROR::COMPUTE_SHADER::LINKING_FAILED\n"
                     << info_log << std::endl;
            return false;
        }
    }

    return true;
}

void upload_texture_async(GLuint texture, const cv::Mat& mat) {
    GLenum format = (mat.channels() == 1) ? GL_RED : GL_BGR;
    GLenum internal_format = (mat.channels() == 1) ? GL_R8 : GL_RGB8;

    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format,
                mat.cols, mat.rows, 0,
                format, GL_UNSIGNED_BYTE, mat.data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void download_texture_async(GLuint texture, cv::Mat& mat) {
    GLenum format = (mat.channels() == 1) ? GL_RED : GL_BGR;

    glBindTexture(GL_TEXTURE_2D, texture);
    glGetTexImage(GL_TEXTURE_2D, 0, format, GL_UNSIGNED_BYTE, mat.data);
}

//=============================================================================
// GPUMorphology Implementation
//=============================================================================

GPUMorphology::GPUMorphology() {
    init();
}

GPUMorphology::~GPUMorphology() {
    cleanup();
}

void GPUMorphology::init() {
    if (initialized_) return;

    // Compile compute shader
    program_ = compile_compute_shader(morphology_shader_source);
    if (program_ == 0) {
        std::cerr << "Failed to compile morphology compute shader" << std::endl;
        return;
    }

    // Create textures (will be resized on first use)
    glGenTextures(1, &input_texture_);
    glGenTextures(1, &output_texture_);

    initialized_ = true;
    std::cout << "GPUMorphology initialized" << std::endl;
}

void GPUMorphology::cleanup() {
    if (!initialized_) return;

    if (program_) glDeleteProgram(program_);
    if (input_texture_) glDeleteTextures(1, &input_texture_);
    if (output_texture_) glDeleteTextures(1, &output_texture_);

    program_ = 0;
    input_texture_ = 0;
    output_texture_ = 0;
    initialized_ = false;
}

void GPUMorphology::process(const cv::Mat& input, cv::Mat& output,
                            Operation op, int kernel_size) {
    if (!initialized_) {
        std::cerr << "GPUMorphology not initialized" << std::endl;
        return;
    }

    if (input.type() != CV_8UC1 || output.type() != CV_8UC1) {
        std::cerr << "GPUMorphology requires CV_8UC1 images" << std::endl;
        return;
    }

    // Resize textures if needed
    if (width_ != input.cols || height_ != input.rows) {
        width_ = input.cols;
        height_ = input.rows;

        upload_texture_async(input_texture_, input);
        upload_texture_async(output_texture_, output);
    }

    // Upload input
    upload_texture_async(input_texture_, input);

    // Bind textures as compute images
    glBindImageTexture(0, input_texture_, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
    glBindImageTexture(1, output_texture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8);

    // Set uniforms
    glUseProgram(program_);
    glUniform1i(glGetUniformLocation(program_, "kernel_size"), kernel_size);
    glUniform1i(glGetUniformLocation(program_, "operation"), static_cast<int>(op));

    // Dispatch compute shader
    GLuint groups_x = (width_ + 15) / 16;
    GLuint groups_y = (height_ + 15) / 16;
    glDispatchCompute(groups_x, groups_y, 1);

    // Wait for completion
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // Download result
    download_texture_async(output_texture_, output);
}

//=============================================================================
// GPUHistogram Implementation
//=============================================================================

GPUHistogram::GPUHistogram() {
    init();
}

GPUHistogram::~GPUHistogram() {
    cleanup();
}

void GPUHistogram::init() {
    if (initialized_) return;

    // Compile compute shader
    program_ = compile_compute_shader(histogram_shader_source);
    if (program_ == 0) {
        std::cerr << "Failed to compile histogram compute shader" << std::endl;
        return;
    }

    // Create texture
    glGenTextures(1, &input_texture_);

    // Create SSBO for histogram
    glGenBuffers(1, &histogram_buffer_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, histogram_buffer_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 256 * sizeof(uint32_t),
                nullptr, GL_DYNAMIC_COPY);

    initialized_ = true;
    std::cout << "GPUHistogram initialized" << std::endl;
}

void GPUHistogram::cleanup() {
    if (!initialized_) return;

    if (program_) glDeleteProgram(program_);
    if (input_texture_) glDeleteTextures(1, &input_texture_);
    if (histogram_buffer_) glDeleteBuffers(1, &histogram_buffer_);

    program_ = 0;
    input_texture_ = 0;
    histogram_buffer_ = 0;
    initialized_ = false;
}

void GPUHistogram::compute(const cv::Mat& input, std::vector<uint32_t>& histogram) {
    if (!initialized_) {
        std::cerr << "GPUHistogram not initialized" << std::endl;
        return;
    }

    if (input.type() != CV_8UC1) {
        std::cerr << "GPUHistogram requires CV_8UC1 image" << std::endl;
        return;
    }

    histogram.resize(256, 0);

    // Clear histogram buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, histogram_buffer_);
    uint32_t zero_data[256] = {0};
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 256 * sizeof(uint32_t), zero_data);

    // Upload input texture
    if (width_ != input.cols || height_ != input.rows) {
        width_ = input.cols;
        height_ = input.rows;
    }
    upload_texture_async(input_texture_, input);

    // Bind texture and buffer
    glBindImageTexture(0, input_texture_, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, histogram_buffer_);

    // Dispatch compute shader
    glUseProgram(program_);
    GLuint groups_x = (width_ + 15) / 16;
    GLuint groups_y = (height_ + 15) / 16;
    glDispatchCompute(groups_x, groups_y, 1);

    // Wait for completion
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Download histogram
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, histogram_buffer_);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                      256 * sizeof(uint32_t), histogram.data());
}

//=============================================================================
// GPUFitnessEvaluator Implementation
//=============================================================================

GPUFitnessEvaluator::GPUFitnessEvaluator() {
    init();
}

GPUFitnessEvaluator::~GPUFitnessEvaluator() {
    cleanup();
}

void GPUFitnessEvaluator::init() {
    if (initialized_) return;

    // Compile compute shader
    program_ = compile_compute_shader(fitness_shader_source);
    if (program_ == 0) {
        std::cerr << "Failed to compile fitness compute shader" << std::endl;
        return;
    }

    // Create texture
    glGenTextures(1, &input_texture_);

    // Create SSBO for metrics (4 floats)
    glGenBuffers(1, &metrics_buffer_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, metrics_buffer_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * sizeof(float),
                nullptr, GL_DYNAMIC_COPY);

    initialized_ = true;
    std::cout << "GPUFitnessEvaluator initialized" << std::endl;
}

void GPUFitnessEvaluator::cleanup() {
    if (!initialized_) return;

    if (program_) glDeleteProgram(program_);
    if (input_texture_) glDeleteTextures(1, &input_texture_);
    if (metrics_buffer_) glDeleteBuffers(1, &metrics_buffer_);

    program_ = 0;
    input_texture_ = 0;
    metrics_buffer_ = 0;
    initialized_ = false;
}

void GPUFitnessEvaluator::evaluate_batch(const std::vector<cv::Mat>& frames,
                                        std::vector<float>& metrics) {
    if (!initialized_) {
        static bool error_logged = false;
        if (!error_logged) {
            std::cerr << "GPUFitnessEvaluator not initialized (GPU acceleration disabled, using CPU fallback)" << std::endl;
            error_logged = true;
        }
        return;
    }

    metrics.clear();

    for (const auto& frame : frames) {
        if (frame.type() != CV_8UC1) {
            std::cerr << "GPUFitnessEvaluator requires CV_8UC1 frames (got type=" << frame.type() << ")" << std::endl;
            continue;
        }

        // Clear metrics buffer
        float zero_data[4] = {0.0f, 0.0f, 0.0f, static_cast<float>(frame.total())};
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, metrics_buffer_);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4 * sizeof(float), zero_data);

        // Upload frame
        upload_texture_async(input_texture_, frame);

        // Bind texture and buffer
        glBindImageTexture(0, input_texture_, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, metrics_buffer_);

        // Dispatch compute shader
        glUseProgram(program_);
        GLuint groups_x = (frame.cols + 15) / 16;
        GLuint groups_y = (frame.rows + 15) / 16;
        glDispatchCompute(groups_x, groups_y, 1);

        // Wait for completion
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Download metrics
        float result[4];
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, metrics_buffer_);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4 * sizeof(float), result);

        // Calculate final metrics
        float total_pixels = result[3];

        // Validate GPU results
        if (total_pixels <= 0.0f || !std::isfinite(result[0]) || !std::isfinite(result[1]) || !std::isfinite(result[2])) {
            std::cerr << "WARNING: GPU produced invalid results, disabling GPU acceleration" << std::endl;
            std::cerr << "  result[0]=" << result[0] << " result[1]=" << result[1]
                     << " result[2]=" << result[2] << " total_pixels=" << total_pixels << std::endl;

            // Disable GPU and mark as not initialized
            cleanup();
            metrics.clear();
            return;
        }

        float mean = result[0] / total_pixels;
        float variance = (result[1] / total_pixels) - (mean * mean);
        float non_zero_ratio = result[2] / total_pixels;

        // Validate calculated values
        if (!std::isfinite(mean) || !std::isfinite(variance) || !std::isfinite(non_zero_ratio)) {
            std::cerr << "WARNING: GPU calculations produced invalid results, disabling GPU acceleration" << std::endl;
            std::cerr << "  mean=" << mean << " variance=" << variance << " non_zero_ratio=" << non_zero_ratio << std::endl;

            cleanup();
            metrics.clear();
            return;
        }

        // Combined fitness metric
        float fitness = mean + variance * 0.5f + non_zero_ratio * 0.3f;
        metrics.push_back(fitness);
    }
}

} // namespace gpu
} // namespace video
