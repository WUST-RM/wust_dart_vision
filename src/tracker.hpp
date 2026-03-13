#pragma once
#include "kf.hpp"
#include "toml++/toml.hpp"
#include "type.hpp"
#include <cstdlib>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <utility>
#include <chrono>
#include <algorithm>
#include <cmath>

namespace dart_vision {

class Tracker {
public:
    using Ptr = std::unique_ptr<Tracker>;

    enum State : int {
        LOST = 0,
        DETECTING,
        TRACKING,
        TEMP_LOST,
    };

    Tracker(const toml::table& config, std::pair<cv::Mat, cv::Mat> camera_info)
        : tracker_state(LOST),
          box_kf_(nullptr),
          pos_kf_(nullptr),
          detect_count_(0),
          lost_count_(0),
          latency_ms_(0.f),
          roi_target_w_(320),
          roi_target_h_(240)
    {
        // create KFs (parameters as before)
        box_kf_ = std::make_unique<CVKalman4x2>(
            0.06f,
            static_cast<float>(config["box_proc_noise_pos"].value_or(0.0)),
            static_cast<float>(config["box_proc_noise_vel"].value_or(0.0)),
            static_cast<float>(config["box_meas_noise"].value_or(1.0))
        );

        pos_kf_ = std::make_unique<CVKalman6x3>(
            0.06f,
            static_cast<float>(config["pos_proc_noise_pos"].value_or(0.0)),
            static_cast<float>(config["pos_proc_noise_vel"].value_or(0.0)),
            static_cast<float>(config["pos_meas_noise"].value_or(1.0))
        );

        tracking_thres_ = static_cast<int>(config["tracking_thres"].value_or(5));
        lost_dt_ = static_cast<float>(config["lost_dt"].value_or(0.05));
        max_region_err_ = static_cast<int>(config["max_region_err"].value_or(2)); // guard default
        roi_target_h_ = static_cast<int>(config["roi_target_h"].value_or(240));
        roi_target_w_ = static_cast<int>(config["roi_target_w"].value_or(320));

        // convert camera_info mats to CV_32F once to avoid later cast issues
        convertCameraInfoToFloat(camera_info);
    }

    static Ptr create(const toml::table& config, std::pair<cv::Mat, cv::Mat> camera_info) {
        return std::make_unique<Tracker>(config, std::move(camera_info));
    }

    // main tracking API called per-frame
    void track(const Lights& lights, const Eigen::Matrix4f& T_camera_to_odom) noexcept {
        // compute dt from timestamps safely
        const float dt = std::max(1e-6f,
            std::chrono::duration<float>(lights.timestamp - last_time_).count());

        // latency (ms)
        const auto now = std::chrono::steady_clock::now();
        latency_ms_ = std::chrono::duration<float, std::milli>(now - lights.timestamp).count();

        last_time_ = lights.timestamp;

        // compute lost threshold (guard dt)
        lost_thres_ = std::max(1, static_cast<int>(std::abs(lost_dt_ / dt)));

        image_size_ = lights.image_size;

        bool found = false;
        if (tracker_state == LOST) {
            found = init(lights, T_camera_to_odom);
        } else {
            found = update(lights, T_camera_to_odom);
        }

        fsm(found);
    }

    // return expanded ROI (keeps same semantics)
    cv::Rect expanded() const noexcept {
        const int TARGET_W = roi_target_w_;
        const int TARGET_H = roi_target_h_;
        const float ASPECT = static_cast<float>(TARGET_W) / static_cast<float>(TARGET_H);
        const int TARGET_AREA = TARGET_W * TARGET_H;

        const cv::Rect2f bbox = bbox_;
        const int bw = bbox.width;
        const int bh = bbox.height;
        if (bw <= 0 || bh <= 0) return cv::Rect();

        // compute scaled region preserving aspect ratio
        int w = bw;
        int h = bh;
        const float b_aspect = static_cast<float>(bw) / static_cast<float>(bh);
        if (b_aspect > ASPECT) {
            h = static_cast<int>(std::round(bw / ASPECT));
        } else {
            w = static_cast<int>(std::round(bh * ASPECT));
        }

        // scale to target area
        const float scale = std::sqrt(static_cast<float>(TARGET_AREA) / static_cast<float>(w * h));
        w = std::min(image_size_.width, static_cast<int>(std::round(w * scale)));
        h = std::min(image_size_.height, static_cast<int>(std::round(h * scale)));

        // final adjust to exact aspect
        if (static_cast<float>(w) / static_cast<float>(h) > ASPECT) {
            h = static_cast<int>(std::round(w / ASPECT));
        } else {
            w = static_cast<int>(std::round(h * ASPECT));
        }

        const int cx = bbox.x + bw / 2;
        const int cy = bbox.y + bh / 2;

        int x = cx - w / 2;
        int y = cy - h / 2;
        x = std::clamp(x, 0, image_size_.width - w);
        y = std::clamp(y, 0, image_size_.height - h);

        cv::Rect rect(x, y, w, h);
        rect &= cv::Rect(0, 0, image_size_.width, image_size_.height);
        return rect;
    }

    cv::Vec6f predict_future(std::chrono::steady_clock::time_point cur_time) const noexcept {
        float dt = std::chrono::duration<float>(cur_time - time_stamp_).count();
        return predict_future(dt);
    }

    cv::Vec6f predict_future(float dt) const noexcept {
        cv::Vec6f s = pos_state_;
        s[0] += dt * s[1];
        s[2] += dt * s[3];
        s[4] += dt * s[5];
        return s;
    }

    void draw(cv::Mat& img) const noexcept {
        if (tracker_state == TRACKING || tracker_state == TEMP_LOST) {
            cv::Point2f pos(box_state_(0), box_state_(2));
            cv::circle(img, pos, 10, cv::Scalar(0, 255, 0), 2);
            float vx = box_state_(1);
            float vy = box_state_(3);
            const float scale = 1.0f;
            cv::Point2f endPoint(pos.x + vx * scale, pos.y + vy * scale);
            cv::arrowedLine(img, pos, endPoint, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.3f);

            // status text: keep concise to avoid heavy string ops each frame
            std::string debugTxt = "x:" + std::to_string(pos_state_[0])
                                 + " vx:" + std::to_string(pos_state_[1])
                                 + " y:" + std::to_string(pos_state_[2])
                                 + " vz:" + std::to_string(pos_state_[4]);
            int baseline = 0;
            cv::Size ts = cv::getTextSize(debugTxt, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Point origin(5, ts.height + 5);
            cv::putText(img, debugTxt, origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1, cv::LINE_AA);
        }

        // latency text (small cost)
        const std::string delay = "Delay: " + std::to_string(static_cast<int>(latency_ms_)) + " ms";
        int baseline = 0;
        cv::Size txt = cv::getTextSize(delay, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        cv::Point origin(img.cols - txt.width - 10, txt.height + 10);
        cv::putText(img, delay, origin, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    }

    bool check() const noexcept {
        if (tracker_state == TRACKING) return true;
        if (tracker_state == TEMP_LOST) {
            const float dt = std::chrono::duration<float>(std::chrono::steady_clock::now() - time_stamp_).count();
            return dt < lost_dt_;
        }
        return false;
    }

private:
    // ---------- internal helpers ----------
    void convertCameraInfoToFloat(const std::pair<cv::Mat, cv::Mat>& cam) noexcept {
        // Expect incoming camera_info as either CV_64F or CV_32F. Convert to CV_32F.
        if (cam.first.empty() || cam.second.empty()) {
            camera_info_.first = cv::Mat();
            camera_info_.second = cv::Mat();
            return;
        }
        cam.first.convertTo(camera_info_.first, CV_32F);
        cam.second.convertTo(camera_info_.second, CV_32F);
    }

    void fsm(bool found) noexcept {
        switch (tracker_state) {
            case LOST:
                if (found) {
                    tracker_state = DETECTING;
                    detect_count_ = 1;
                }
                break;
            case DETECTING:
                if (found) {
                    if (++detect_count_ > tracking_thres_) {
                        detect_count_ = 0;
                        tracker_state = TRACKING;
                    }
                } else {
                    detect_count_ = 0;
                    tracker_state = LOST;
                }
                break;
            case TRACKING:
                if (!found) {
                    tracker_state = TEMP_LOST;
                    lost_count_ = 1;
                }
                break;
            case TEMP_LOST:
                if (!found) {
                    if (++lost_count_ > lost_thres_) {
                        lost_count_ = 0;
                        tracker_state = LOST;
                    }
                } else {
                    lost_count_ = 0;
                    tracker_state = TRACKING;
                }
                break;
            default:
                tracker_state = LOST;
                break;
        }
    }

    bool init(const Lights& lights, const Eigen::Matrix4f& T_camera_to_odom) noexcept {
        int best = -1;
        float max_area = -1.f;
        const auto& vec = lights.lights;
        for (size_t i = 0; i < vec.size(); ++i) {
            const float a = vec[i].area;
            if (a > max_area) { max_area = a; best = static_cast<int>(i); }
        }
        if (best < 0) return false;

        tracker_state = DETECTING;
        box_kf_->reset();
        const auto& c = lights.lights[best].center;
        box_kf_->setInitialState(c.x, 0.f, c.y, 0.f);

        const Eigen::Vector3f pos = lights.lights[best].getPos(camera_info_, T_camera_to_odom).cast<float>();
        time_stamp_ = lights.timestamp;
        pos_kf_->reset();
        pos_kf_->setInitialState(pos.x(), 0.f, pos.y(), 0.f, pos.z(), 0.f);

        bbox_ = lights.lights[best].bbox;
        return true;
    }

    bool update(const Lights& lights, const Eigen::Matrix4f& T_camera_to_odom) noexcept {
        const float dt = std::max(1e-6f, std::chrono::duration<float>(lights.timestamp - time_stamp_).count());
        box_kf_->setDt(dt);
        pos_kf_->setDt(dt);

        box_state_ = box_kf_->predict();
        pos_state_ = pos_kf_->predict();

        const auto& vec = lights.lights;
        int best = -1;
        float minErr = std::numeric_limits<float>::max();
        const float maxErr = static_cast<float>(image_size_.width) / std::max(1, max_region_err_);

        for (size_t i = 0; i < vec.size(); ++i) {
            const float err = cv::norm(vec[i].center - cv::Point2f(box_state_[0], box_state_[2]));
            if (err < minErr && err < maxErr) {
                minErr = err;
                best = static_cast<int>(i);
            }
        }
        if (best < 0) return false;

        const Eigen::Vector3f pos = vec[best].getPos(camera_info_, T_camera_to_odom).cast<float>();
        pos_state_ = pos_kf_->correct(pos.x(), pos.y(), pos.z());
        box_state_ = box_kf_->correct(vec[best].center);

        time_stamp_ = lights.timestamp;
        bbox_ = vec[best].bbox;
        return true;
    }

    // ---------- members ----------
public:
    State tracker_state;

private:
    std::unique_ptr<CVKalman4x2> box_kf_;
    std::unique_ptr<CVKalman6x3> pos_kf_;

    int tracking_thres_{5};
    int lost_thres_{1};
    int detect_count_{0};
    int lost_count_{0};
    float lost_dt_{0.05f};
    int max_region_err_{2};

    std::chrono::steady_clock::time_point last_time_;
    std::chrono::steady_clock::time_point time_stamp_;

    cv::Vec4f box_state_;
    cv::Vec6f pos_state_;
    cv::Size image_size_{0,0};
    cv::Rect2f bbox_;
    float latency_ms_{0.f};
    int roi_target_w_{320};
    int roi_target_h_{240};

    // camera info stored as float mats
    std::pair<cv::Mat, cv::Mat> camera_info_;
};

} // namespace dart_vision