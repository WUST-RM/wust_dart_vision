#pragma once
#include "kf.hpp"
#include "toml++/toml.hpp"
#include "type.hpp"
#include <cstdlib>
#include <memory>
#include <opencv2/core/matx.hpp>
#include <string>
#include <utility>
namespace dart_vision {
class Tracker {
public:
    using Ptr = std::unique_ptr<Tracker>;
    enum State {
        LOST,
        DETECTING,
        TRACKING,
        TEMP_LOST,
    } tracker_state = LOST;
    Tracker(const toml::table& config, std::pair<cv::Mat, cv::Mat> camera_info) {
        box_kf_ = std::make_unique<CVKalman4x2>(
            0.06,
            config["box_proc_noise_pos"].value_or(0.0),
            config["box_proc_noise_vel"].value_or(0.0),
            config["box_meas_noise"].value_or(1.0)
        );
        dis_kf_ = std::make_unique<CVKalman2x1>(
            0.06,
            config["dis_proc_noise_pos"].value_or(0.0),
            config["dis_proc_noise_vel"].value_or(0.0),
            config["dis_meas_noise"].value_or(1.0)
        );
        tracking_thres_ = config["tracking_thres"].value_or(5);
        lost_dt_ = config["lost_dt"].value_or(0.05);
        max_region_err_ = config["max_region_err"].value_or(0.5);
        roi_target_h_ = config["roi_target_h"].value_or(240);
        roi_target_w_ = config["roi_target_w"].value_or(320);
        camera_info_ = camera_info;
    }
    static Ptr create(const toml::table& config, std::pair<cv::Mat, cv::Mat> camera_info) {
        return std::make_unique<Tracker>(config, camera_info);
    }
    void fsm(bool found) noexcept {
        if (tracker_state == DETECTING) {
            if (found) {
                detect_count_++;
                if (detect_count_ > tracking_thres_) {
                    detect_count_ = 0;
                    tracker_state = TRACKING;
                }
            } else {
                detect_count_ = 0;
                tracker_state = LOST;
            }
        } else if (tracker_state == TRACKING) {
            if (!found) {
                tracker_state = TEMP_LOST;
                lost_count_++;
            }
        } else if (tracker_state == TEMP_LOST) {
            if (!found) {
                lost_count_++;
                if (lost_count_ > lost_thres_) {
                    lost_count_ = 0;
                    tracker_state = LOST;
                }
            } else {
                tracker_state = TRACKING;
                lost_count_ = 0;
            }
        }
    }
    void track(const Lights& lights) noexcept {
        const float dt = std::chrono::duration<float>(lights.timestamp - last_time_).count();
        const auto now = std::chrono::steady_clock::now();

        latency_ms_ = std::chrono::duration<float, std::milli>(now - lights.timestamp).count();
        last_time_ = lights.timestamp;
        lost_thres_ = std::abs(static_cast<int>(lost_dt_ / dt));
        bool found;
        image_size_ = lights.image_size;
        if (tracker_state == LOST) {
            found = init(lights);
        } else {
            found = update(lights);
        }
        fsm(found);
    }
    bool init(const Lights& lights) noexcept {
        int best_id = -1;
        float max_score = -1e9;
        for (int i = 0; i < lights.lights.size(); i++) {
            if (lights.lights[i].area > max_score) {
                max_score = lights.lights[i].area;
                best_id = i;
            }
        }
        if (best_id == -1) {
            return false;
        }
        tracker_state = DETECTING;
        box_kf_->reset();
        box_kf_->setInitialState(
            lights.lights[best_id].center.x,
            0.0f,
            lights.lights[best_id].center.y,
            0.0f
        );
        auto tvec = lights.lights[best_id].getTvec(camera_info_);
        float dis = cv::norm(tvec);
        dis_kf_->reset();
        dis_kf_->setInitialState(dis, 0.0f);
        time_stamp_ = lights.timestamp;
        bbox_ = lights.lights[best_id].bbox;
        std::cout << "tracker init " << std::endl;
        return true;
    }
    cv::Rect expanded() const noexcept {
        const int TARGET_W = roi_target_w_;
        const int TARGET_H = roi_target_h_;
        const float ASPECT = static_cast<float>(TARGET_W) / TARGET_H;
        const int TARGET_AREA = TARGET_W * TARGET_H;

        const auto bbox = bbox_;

        const int bw = bbox.width;
        const int bh = bbox.height;

        if (bw <= 0 || bh <= 0)
            return {};

        int w = bw;
        int h = bh;

        if (static_cast<float>(bw) / bh > ASPECT)
            h = static_cast<int>(bw / ASPECT);
        else
            w = static_cast<int>(bh * ASPECT);

        const float scale = std::sqrt(static_cast<float>(TARGET_AREA) / (w * h));
        w = static_cast<int>(w * scale);
        h = static_cast<int>(h * scale);

        if (static_cast<float>(w) / h > ASPECT)
            h = static_cast<int>(w / ASPECT);
        else
            w = static_cast<int>(h * ASPECT);

        w = std::min(w, image_size_.width);
        h = std::min(h, image_size_.height);

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
    std::pair<cv::Vec4f, cv::Vec2f> predict_future(std::chrono::steady_clock::time_point cur_time
    ) const noexcept {
        float dt = std::chrono::duration<float>(cur_time - time_stamp_).count();

        return predict_future(dt);
    }
    std::pair<cv::Vec4f, cv::Vec2f> predict_future(float dt) const noexcept {
        cv::Vec4f box_state = box_state_;
        box_state[0] += dt * box_state[1];
        box_state[2] += dt * box_state[3];
        cv::Vec2f dis_state = dis_state_;
        dis_state[0] += dt * dis_state[1];
        return std::make_pair(box_state, dis_state);
    }
    bool update(const Lights& lights) noexcept {
        const float dt = std::chrono::duration<float>(lights.timestamp - time_stamp_).count();
        box_kf_->setDt(dt);
        dis_kf_->setDt(dt);
        box_state_ = box_kf_->predict();
        dis_state_ = dis_kf_->predict();
        int best_id = -1;
        float min_error = 1e9;
        for (int i = 0; i < lights.lights.size(); i++) {
            float centor_error =
                cv::norm(lights.lights[i].center - cv::Point2f(box_state_[0], box_state_[2]));
            if (centor_error < min_error
                && centor_error < static_cast<float>(image_size_.width) / max_region_err_)
            {
                min_error = centor_error;
                best_id = i;
            }
        }
        if (best_id == -1) {
            return false;
        }
        auto tvec = lights.lights[best_id].getTvec(camera_info_);
        float dis = cv::norm(tvec);

            dis_state_ = dis_kf_->correct(dis);
        
        box_state_ = box_kf_->correct(lights.lights[best_id].center);

        time_stamp_ = lights.timestamp;
        bbox_ = lights.lights[best_id].bbox;
        return true;
    }
    void draw(cv::Mat& img) const noexcept {
        if (tracker_state == TRACKING || tracker_state == TEMP_LOST) {
            cv::Point2f pos(box_state_(0), box_state_(2));
            cv::circle(img, pos, 10, cv::Scalar(0, 255, 0), 2);

            float vx = box_state_(1);
            float vy = box_state_(3);
            float scale = 1.0f;

            cv::Point2f endPoint(pos.x + vx * scale, pos.y + vy * scale);
            cv::arrowedLine(img, pos, endPoint, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.3);
            cv::putText(
                img,
                ("dis: " + std::to_string(dis_state_[0])),
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 0, 255),
                2
            );
        }

        const std::string text = "Delay: " + std::to_string(latency_ms_) + " ms";

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);

        cv::Point origin(img.cols - textSize.width - 10, textSize.height + 10);

        cv::putText(
            img,
            text,
            origin,
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA
        );
    }

    bool check() const noexcept {
        return tracker_state == TRACKING
            || tracker_state == TEMP_LOST
            && std::chrono::duration<float>(std::chrono::steady_clock::now() - time_stamp_).count()
                < lost_dt_;
    }

    std::unique_ptr<CVKalman4x2> box_kf_;
    std::unique_ptr<CVKalman2x1> dis_kf_;
    int tracking_thres_;
    int lost_thres_;
    int detect_count_ = 0;
    int lost_count_ = 0;
    float lost_dt_;
    int max_region_err_;
    std::chrono::steady_clock::time_point last_time_;
    std::chrono::steady_clock::time_point time_stamp_;
    cv::Vec4f box_state_;
    cv::Vec2f dis_state_;
    cv::Size image_size_;
    cv::Rect2f bbox_;
    double latency_ms_;
    int roi_target_w_ = 0;
    int roi_target_h_ = 0;
    std::pair<cv::Mat, cv::Mat> camera_info_;
};
} // namespace dart_vision