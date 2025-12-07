#pragma once
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>
namespace dart_vision {
struct Frame {
    cv::Mat image;
    int id;
    std::chrono::steady_clock::time_point timestamp;
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
struct Params {
    int image_width = 640;
    int image_height = 480;
    int exposure = 10;
    int D_thresh = 11;
    float one_one_diff = 0.3;
    int scale_percent = 20;
    std::string device_name = "/dev/video0";
    float meas_noise = 1.0f;
    float proc_noise_pos = 1e-2f;
    float proc_noise_vel = 1e-3f;
    int tracking_thres = 5;
    float lost_dt = 0.5f;
    int max_region_err = 5;
    bool use_serial = false;
    std::string serial_device = "/dev/ttyACM0";
    bool use_saver = false;
    int roi_target_w = 352;
    int roi_target_h = 288;
    void load(const std::string& path) {
        YAML::Node config = YAML::LoadFile(path);
        image_width = config["image_width"].as<int>();
        image_height = config["image_height"].as<int>();
        exposure = config["exposure"].as<int>();
        D_thresh = config["D_thresh"].as<int>();
        one_one_diff = config["one_one_diff"].as<float>();
        scale_percent = config["scale_percent"].as<int>();
        device_name = config["device_name"].as<std::string>();
        meas_noise = config["meas_noise"].as<float>();
        proc_noise_pos = config["proc_noise_pos"].as<float>();
        proc_noise_vel = config["proc_noise_vel"].as<float>();
        tracking_thres = config["tracking_thres"].as<int>();
        lost_dt = config["lost_dt"].as<float>();
        max_region_err = config["max_region_err"].as<float>();
        use_serial = config["use_serial"].as<bool>();
        serial_device = config["serial_device"].as<std::string>();
        use_saver = config["use_saver"].as<bool>();
        roi_target_w = config["roi_target_w"].as<int>();
        roi_target_h = config["roi_target_h"].as<int>();
    }
};
} // namespace dart_vision
