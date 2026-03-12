#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
namespace dart_vision {
struct Frame {
    cv::Mat image;
    std::chrono::steady_clock::time_point timestamp;
    cv::Point2f offset;
    cv::Rect expanded;
};
constexpr float half = 0.1f / 2.0f; // 0.05
static std::vector<cv::Point3f> object_points = {
    { -half, -half, 0.0f }, // 左上
    { half, -half, 0.0f }, // 右上
    { half, half, 0.0f }, // 右下
    { -half, half, 0.0f } // 左下
};

struct Light {
    cv::Point2f center;
    cv::Rect2f bbox;
    std::chrono::steady_clock::time_point timestamp;
    bool valid = false;
    float area;

    std::vector<cv::Point2f> toCorners() const {
        std::vector<cv::Point2f> corners;
        corners.push_back(cv::Point2f(bbox.x, bbox.y)); // 左上
        corners.push_back(cv::Point2f(bbox.x + bbox.width, bbox.y)); // 右上
        corners.push_back(cv::Point2f(bbox.x + bbox.width, bbox.y + bbox.height)); // 右下
        corners.push_back(cv::Point2f(bbox.x, bbox.y + bbox.height)); // 左下
        return corners;
    }
    cv::Mat getTvec(const std::pair<cv::Mat, cv::Mat>& camera_info) const {
        auto lm = toCorners();
        cv::Mat rvec, tvec;
        cv::solvePnP(
            object_points,
            lm,
            camera_info.first,
            camera_info.second,
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_IPPE
        );
        return tvec;
    }
};
struct Lights {
    std::chrono::steady_clock::time_point timestamp;
    std::vector<Light> lights;
    cv::Size2f image_size;
};
struct SerialFrame {
    float pitch;
    float yaw;
    float dis;
    uint32_t sum;
} __attribute__((packed));
} // namespace dart_vision