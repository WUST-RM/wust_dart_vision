#pragma once
#include "type.hpp"
#include "utils.hpp"
#include <toml++/toml.h>

namespace dart_vision {
class LightDetector {
public:
    using Ptr = std::unique_ptr<LightDetector>;
    LightDetector(const toml::table& config) {
        D_threshold_ = config["D_threshold"].value_or(10);
        one_one_diff_ = config["one_one_diff"].value_or(0.5f);
        min_area_ = config["min_area"].value_or(100.0f);
    }
    static Ptr create(const toml::table& config) {
        return std::make_unique<LightDetector>(config);
    }
    cv::Mat preProcess(const cv::Mat& roi) const noexcept {
        cv::Mat mask;
#ifdef X86
        static cv::Mat channels[3];
        static cv::Mat maxRB;
        static cv::Mat diff;
        cv::split(roi, channels); // B G R

        cv::max(channels[2], channels[0], maxRB); // max(R,B)
        cv::subtract(channels[1], maxRB, diff); // G - max(R,B)

        cv::threshold(diff, mask, D_threshold_, 255, cv::THRESH_BINARY);
        // mask = cv::Mat(roi.rows, roi.cols, CV_8UC1);
        // sse_diff_threshold_bgr(roi, mask, static_cast<uint8_t>(D_threshold_));
#endif
#ifdef ARM
        mask = cv::Mat(roi.rows, roi.cols, CV_8UC1);
        neon_diff_threshold_bgr(roi, mask, static_cast<uint8_t>(D_threshold_));
#endif
        return mask;
    }
    void process(const cv::Mat& mask, Lights& lights, Frame& frame, bool need_draw) const noexcept {
        std::vector<std::vector<cv::Point>> contours;
        contours.reserve(100);
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        lights.lights.reserve(contours.size());

        for (const auto& c: contours) {
            const cv::Moments m = cv::moments(c);
            const float area = m.m00;
            if (area < min_area_)
                continue;

            cv::Rect box = cv::boundingRect(c);
            const float ratio = float(box.width) / float(box.height);
            if (ratio < 1 - one_one_diff_ || ratio > 1 + one_one_diff_)
                continue;

            if (area <= 0.0f)
                continue;

            cv::Point2f center(
                static_cast<float>(m.m10 / m.m00),
                static_cast<float>(m.m01 / m.m00)
            );

            cv::Rect2f rect(box.x + frame.offset.x, box.y + frame.offset.y, box.width, box.height);

            lights.lights.emplace_back(Light {
                .center = center + frame.offset,
                .bbox = rect,
                .timestamp = frame.timestamp,
                .valid = true,
                .area = area,
            });

            if (need_draw) {
                cv::rectangle(frame.image, rect, cv::Scalar(255, 0, 0), 2);
                cv::circle(frame.image, center + frame.offset, 3, cv::Scalar(0, 255, 0), -1);
            }
        }
    }
    Lights detect(Frame& frame, bool need_show, bool need_draw) noexcept {
        auto& image = frame.image;

        Lights lights;
        lights.image_size = image.size();
        lights.timestamp = frame.timestamp;

        cv::Mat roi = image(frame.expanded);
        const auto offset = frame.offset;

        const cv::Mat mask = preProcess(roi);
#ifdef X86
        static bool window_created = false;
        if (!window_created) {
            cv::namedWindow("mask", cv::WINDOW_NORMAL);
            window_created = true;
            cv::createTrackbar("D_thresh %", "mask", &D_threshold_, 255);
        }
        if (need_show) {
            cv::imshow("mask", mask);
            cv::waitKey(1);
        }
#endif
        process(mask, lights, frame, need_draw);

        return lights;
    }
    int D_threshold_;
    float one_one_diff_;
    float min_area_;
};
} // namespace dart_vision