#pragma once

#include <opencv2/opencv.hpp>
#include <GL/glew.h>

namespace video {

/**
 * OpenGL texture manager for video frames
 *
 * Handles GPU texture creation, uploading, and lifecycle management.
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
     * Get the last uploaded frame (CPU copy)
     * @return Last frame, or empty Mat if none available
     */
    cv::Mat get_last_frame() const { return last_frame_.clone(); }

    /**
     * Reset texture (deletes OpenGL texture)
     */
    void reset();

private:
    void ensure_texture_created();

    GLuint texture_id_ = 0;
    int width_ = 0;
    int height_ = 0;
    cv::Mat last_frame_;  // Keep CPU copy for capture
};

} // namespace video
