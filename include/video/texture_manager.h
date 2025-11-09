#pragma once

#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include "video/frame_ref.h"

namespace video {

/**
 * OpenGL texture manager for video frames (ZERO-COPY optimized)
 *
 * Handles GPU texture creation, uploading, and lifecycle management.
 * Uses FrameRef to eliminate unnecessary frame cloning.
 */
class TextureManager {
public:
    TextureManager();
    ~TextureManager();

    // Non-copyable
    TextureManager(const TextureManager&) = delete;
    TextureManager& operator=(const TextureManager&) = delete;

    /**
     * Upload RGB frame to GPU texture
     * @param rgb_frame Frame in RGB format (will be converted if BGR)
     */
    void upload_frame(const cv::Mat& rgb_frame);

    /**
     * Upload frame from FrameRef (zero-copy)
     * @param frame_ref Frame reference (shares data, no clone)
     */
    void upload_frame(const FrameRef& frame_ref);

    /**
     * Get OpenGL texture ID for rendering
     * @return Texture ID (0 if not yet created)
     */
    GLuint get_texture_id() const { return texture_id_; }

    /**
     * Get current texture width
     * @return Width in pixels
     */
    int get_width() const { return width_; }

    /**
     * Get current texture height
     * @return Height in pixels
     */
    int get_height() const { return height_; }

    /**
     * Get the last uploaded frame (ZERO-COPY - returns FrameRef)
     * @return Last frame reference, or empty if none available
     */
    FrameRef get_last_frame() const { return last_frame_; }

    /**
     * Reset texture (deletes OpenGL texture)
     */
    void reset();

private:
    void ensure_texture_created();

    GLuint texture_id_ = 0;
    int width_ = 0;
    int height_ = 0;
    FrameRef last_frame_;  // Keep CPU copy for capture (zero-copy)
};

} // namespace video
