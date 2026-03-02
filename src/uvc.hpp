#pragma once
#include <opencv2/opencv.hpp>
#include <toml++/toml.h>
namespace dart_vision {
class UVC {
public:
    using Ptr = std::unique_ptr<UVC>;
    UVC(const toml::table& config) {
        std::cout << "try get device_name by ls -l /dev/v4l/by-id/" << std::endl;
        device_name_ = config["device_name"].value_or("");
        fps_ = config["fps"].value_or(30);
        width_ = config["width"].value_or(640);
        height_ = config["height"].value_or(480);
        exposure_ = config["exposure"].value_or(0.0);
        gain_ = config["gain"].value_or(0.0);
        gamma_ = config["gamma"].value_or(0.0);
        std::cout << "UVC loaded: " << device_name_ << " " << width_ << "x" << height_ << "@"
                  << fps_ << std::endl;
        if (!cap_.open(device_name_, cv::CAP_V4L2)) {
            if (!tryCommonDevice()) {
                throw std::runtime_error("Cannot open UVC device: " + device_name_);
            }
        }
        cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
        cap_.set(cv::CAP_PROP_EXPOSURE, exposure_);
        cap_.set(cv::CAP_PROP_GAIN, gain_);
        cap_.set(cv::CAP_PROP_GAMMA, gamma_);
        cap_.set(cv::CAP_PROP_FPS, fps_);
        cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }
    static Ptr create(const toml::table& config) {
        return std::make_unique<UVC>(config);
    }
    cv::VideoCapture& cap() {
        return cap_;
    }
    cv::Mat read() {
        cv::Mat frame;

        if (!cap_.grab())
            return frame;

        cap_.retrieve(frame);

        last_frame_time_ = std::chrono::steady_clock::now();
        return frame;
    }
    bool tryCommonDevice() {
        for (int i = 0; i < 10; i++) {
            std::string device_name = "/dev/video" + std::to_string(i);
            if (cap_.open(device_name, cv::CAP_V4L2)) {
                device_name_ = device_name;
                std::cout << "UVC try open: " << device_name_ << "succes" << std::endl;
                return true;
            }
        }
        return false;
    }
    std::string device_name_;
    int fps_;
    int width_;
    int height_;
    double exposure_;
    double gain_;
    double gamma_;
    cv::VideoCapture cap_;
    std::chrono::steady_clock::time_point last_frame_time_;
};
} // namespace dart_vision