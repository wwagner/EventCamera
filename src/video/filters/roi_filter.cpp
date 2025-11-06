#include "video/filters/roi_filter.h"

namespace video {

void ROIFilter::set_roi(int x, int y, int width, int height) {
    std::lock_guard<std::mutex> lock(mutex_);
    x_ = x;
    y_ = y;
    width_ = width;
    height_ = height;
}

void ROIFilter::set_crop_to_roi(bool crop) {
    std::lock_guard<std::mutex> lock(mutex_);
    crop_to_roi_ = crop;
}

void ROIFilter::set_show_rectangle(bool show) {
    std::lock_guard<std::mutex> lock(mutex_);
    show_rectangle_ = show;
}

cv::Mat ROIFilter::apply(const cv::Mat& input) {
    if (input.empty()) {
        return input;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (!enabled_) {
        return input;
    }

    cv::Mat output;

    if (crop_to_roi_) {
        // Crop to ROI region only
        // Ensure ROI is within bounds
        int x = std::max(0, std::min(x_, input.cols - 1));
        int y = std::max(0, std::min(y_, input.rows - 1));
        int w = std::min(width_, input.cols - x);
        int h = std::min(height_, input.rows - y);

        if (w > 0 && h > 0) {
            cv::Rect roi_rect(x, y, w, h);
            output = input(roi_rect).clone();
        } else {
            output = input.clone();
        }
    } else {
        output = input.clone();

        // Draw ROI rectangle if enabled
        if (show_rectangle_) {
            // Draw bright green rectangle
            cv::rectangle(output,
                         cv::Point(x_, y_),
                         cv::Point(x_ + width_, y_ + height_),
                         cv::Scalar(0, 255, 0), 2);

            // Draw corner markers
            int marker_size = 10;
            cv::line(output,
                    cv::Point(x_, y_),
                    cv::Point(x_ + marker_size, y_),
                    cv::Scalar(0, 255, 0), 3);
            cv::line(output,
                    cv::Point(x_, y_),
                    cv::Point(x_, y_ + marker_size),
                    cv::Scalar(0, 255, 0), 3);
        }
    }

    return output;
}

void ROIFilter::set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
}

bool ROIFilter::is_enabled() const {
    return enabled_;
}

std::string ROIFilter::name() const {
    return "ROI";
}

std::string ROIFilter::description() const {
    return "Region of Interest visualization - crop or highlight specific area";
}

} // namespace video
