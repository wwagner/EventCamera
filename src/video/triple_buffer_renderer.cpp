#include "video/triple_buffer_renderer.h"
#include <chrono>
#include <iostream>

namespace video {

TripleBufferRenderer::TripleBufferRenderer() {
    // Buffers will be created lazily on first frame
}

TripleBufferRenderer::~TripleBufferRenderer() {
    reset();
}

void TripleBufferRenderer::create_texture(GLuint& texture_id) {
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void TripleBufferRenderer::create_pbo(GLuint& pbo_id, size_t size) {
    glGenBuffers(1, &pbo_id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
    // GL_STREAM_DRAW for data that changes every frame
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void TripleBufferRenderer::ensure_gl_resources_created(int width, int height) {
    if (initialized_ && width == width_ && height == height_) {
        return;  // Resources already created for this size
    }

    // Clean up old resources if size changed
    if (initialized_) {
        reset();
    }

    width_ = width;
    height_ = height;

    // Calculate buffer size (RGB, 3 bytes per pixel)
    size_t buffer_size = width * height * 3;

    // Create textures and PBOs for all 3 buffers
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        create_texture(buffers_[i].texture);
        create_pbo(buffers_[i].pbo, buffer_size);

        // Pre-allocate texture storage
        glBindTexture(GL_TEXTURE_2D, buffers_[i].texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    initialized_ = true;
}

void TripleBufferRenderer::async_upload_frame(BufferSlot& slot) {
    if (slot.frame.empty()) {
        return;
    }

    // Get read-only access to frame data
    ReadGuard guard(slot.frame);
    const cv::Mat& frame = guard.get();

    // Ensure resources created
    ensure_gl_resources_created(frame.cols, frame.rows);

    // Convert BGR to RGB if needed
    cv::Mat rgb_frame;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
    } else {
        rgb_frame = frame;
    }

    // Bind PBO for async upload
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, slot.pbo);

    // Map PBO buffer for writing (invalidate old data for performance)
    glBufferData(GL_PIXEL_UNPACK_BUFFER, rgb_frame.total() * rgb_frame.elemSize(),
                 nullptr, GL_STREAM_DRAW);
    void* ptr = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);

    if (ptr) {
        // Copy frame data to PBO
        memcpy(ptr, rgb_frame.data, rgb_frame.total() * rgb_frame.elemSize());
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

        // Upload from PBO to texture (async - GPU DMA transfer)
        glBindTexture(GL_TEXTURE_2D, slot.texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, rgb_frame.cols, rgb_frame.rows,
                        GL_RGB, GL_UNSIGNED_BYTE, 0);  // 0 = use bound PBO

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void TripleBufferRenderer::submit_frame(const FrameRef& frame_ref) {
    if (frame_ref.empty()) {
        return;
    }

    int idx = write_idx_.load(std::memory_order_acquire);

    // Store frame in write buffer (zero-copy share)
    buffers_[idx].frame = frame_ref;
    buffers_[idx].timestamp.store(
        std::chrono::steady_clock::now().time_since_epoch().count(),
        std::memory_order_release);
    buffers_[idx].ready.store(true, std::memory_order_release);
}

void TripleBufferRenderer::submit_frame(FrameRef&& frame_ref) {
    if (frame_ref.empty()) {
        return;
    }

    int idx = write_idx_.load(std::memory_order_acquire);

    // Store frame in write buffer (zero-copy move)
    buffers_[idx].frame = std::move(frame_ref);
    buffers_[idx].timestamp.store(
        std::chrono::steady_clock::now().time_since_epoch().count(),
        std::memory_order_release);
    buffers_[idx].ready.store(true, std::memory_order_release);
}

void TripleBufferRenderer::update() {
    // Check if write buffer has new frame ready
    int write_idx = write_idx_.load(std::memory_order_acquire);
    int upload_idx = upload_idx_.load(std::memory_order_acquire);

    if (buffers_[write_idx].ready.load(std::memory_order_acquire)) {
        // Swap write and upload buffers atomically
        write_idx_.store(upload_idx, std::memory_order_release);
        upload_idx_.store(write_idx, std::memory_order_release);
        buffers_[write_idx].ready.store(false, std::memory_order_release);

        // Now upload from new upload buffer
        async_upload_frame(buffers_[upload_idx]);

        // Swap upload and display buffers for next render
        int display_idx = display_idx_.load(std::memory_order_acquire);
        upload_idx_.store(display_idx, std::memory_order_release);
        display_idx_.store(upload_idx, std::memory_order_release);
    }
}

GLuint TripleBufferRenderer::get_texture_id() const {
    if (!initialized_) {
        return 0;
    }

    int idx = display_idx_.load(std::memory_order_acquire);
    return buffers_[idx].texture;
}

void TripleBufferRenderer::reset() {
    if (!initialized_) {
        return;
    }

    // Delete all OpenGL resources
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        if (buffers_[i].texture != 0) {
            glDeleteTextures(1, &buffers_[i].texture);
            buffers_[i].texture = 0;
        }
        if (buffers_[i].pbo != 0) {
            glDeleteBuffers(1, &buffers_[i].pbo);
            buffers_[i].pbo = 0;
        }
        buffers_[i].frame.reset();
        buffers_[i].ready.store(false);
    }

    width_ = 0;
    height_ = 0;
    initialized_ = false;
}

} // namespace video
