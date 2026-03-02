#pragma once
#include <opencv2/opencv.hpp>
#include <string>
namespace dart_vision {
struct Frame {
    cv::Mat image;
    std::chrono::steady_clock::time_point timestamp;
    cv::Point2f offset;
    cv::Rect expanded;
};
struct Light {
    cv::Point2f center;
    cv::Rect2f bbox;
    std::chrono::steady_clock::time_point timestamp;
    bool valid = false;
    float area;
};
struct Lights {
    std::chrono::steady_clock::time_point timestamp;
    std::vector<Light> lights;
    cv::Size2f image_size;
};
} // namespace dart_vision