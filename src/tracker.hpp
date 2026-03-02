#pragma once
#include "kf.hpp"
#include "toml++/toml.hpp"
#include "type.hpp"
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
    Tracker(const toml::table& config) {
        kf_ = std::make_unique<CVKalman4x2>(
            0.06,
            config["proc_noise_pos"].value_or(0.0),
            config["proc_noise_vel"].value_or(0.0),
            config["meas_noise"].value_or(1.0)
        );
        tracking_thres_ = config["tracking_thres"].value_or(5);
        lost_dt_ = config["lost_dt"].value_or(0.05);
        max_region_err_ = config["max_region_err"].value_or(0.5);
        roi_target_h_ = config["roi_target_h"].value_or(240);
        roi_target_w_ = config["roi_target_w"].value_or(320);
    }
    static Ptr create(const toml::table& config) {
        return std::make_unique<Tracker>(config);
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
        float dt = std::chrono::duration<float>(lights.timestamp - last_time_).count();
        auto now = std::chrono::steady_clock::now();

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
        kf_->reset();
        kf_->setInitialState(
            lights.lights[best_id].center.x,
            0.0f,
            lights.lights[best_id].center.y,
            0.0f
        );
        time_stamp_ = lights.timestamp;
        bbox_ = lights.lights[best_id].bbox;
        std::cout << "tracker init " << std::endl;
        return true;
    }
    cv::Rect expanded() const noexcept {
        const int TARGET_W = roi_target_w_;
        const int TARGET_H = roi_target_h_;
        const float ASPECT = (float)TARGET_W / TARGET_H;
        const int TARGET_AREA = TARGET_W * TARGET_H;

        auto bbox = bbox_;
        int bw = bbox.width, bh = bbox.height;

        int w = bw, h = bh;
        if ((float)bw / bh > ASPECT)
            h = (int)(bw / ASPECT);
        else
            w = (int)(bh * ASPECT);

        float scale = std::sqrt((float)TARGET_AREA / (w * h));
        w = (int)(w * scale);
        h = (int)(h * scale);

        if ((float)w / h > ASPECT)
            h = (int)(w / ASPECT);
        else
            w = (int)(h * ASPECT);

        int cx = bbox.x + bw / 2;
        int cy = bbox.y + bh / 2;

        int x = cx - w / 2;
        int y = cy - h / 2;
        x = std::clamp(x, 0, image_size_.width - w);
        y = std::clamp(y, 0, image_size_.height - h);
        return cv::Rect(x, y, w, h);
    }
    cv::Vec4f predict_future(std::chrono::steady_clock::time_point cur_time) const noexcept {
        float dt = std::chrono::duration<float>(cur_time - time_stamp_).count();

        return predict_future(dt);
    }
    cv::Vec4f predict_future(float dt) const noexcept {
        cv::Vec4f state = state_;
        state[0] += dt * state[2];
        state[1] += dt * state[3];
        return state;
    }
    bool update(const Lights& lights) noexcept {
        float dt = std::chrono::duration<float>(lights.timestamp - time_stamp_).count();
        kf_->setDt(dt);
        state_ = kf_->predict();
        int best_id = -1;
        float min_error = 1e9;
        for (int i = 0; i < lights.lights.size(); i++) {
            float centor_error =
                cv::norm(lights.lights[i].center - cv::Point2f(state_[0], state_[2]));
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
        state_ = kf_->correct(lights.lights[best_id].center);
        time_stamp_ = lights.timestamp;
        bbox_ = lights.lights[best_id].bbox;
        return true;
    }
    void draw(cv::Mat& img) const noexcept {
        if (tracker_state == TRACKING || tracker_state == TEMP_LOST) {
            cv::Point2f pos(state_(0), state_(2));
            cv::circle(img, pos, 10, cv::Scalar(0, 255, 0), 2);

            float vx = state_(1);
            float vy = state_(3);
            float scale = 1.0f;

            cv::Point2f endPoint(pos.x + vx * scale, pos.y + vy * scale);
            cv::arrowedLine(img, pos, endPoint, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.3);
        }

        std::string text = "Delay: " + std::to_string(latency_ms_) + " ms";

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

    std::unique_ptr<CVKalman4x2> kf_;
    int tracking_thres_;
    int lost_thres_;
    int detect_count_ = 0;
    int lost_count_ = 0;
    float lost_dt_;
    int max_region_err_;
    std::chrono::steady_clock::time_point last_time_;
    std::chrono::steady_clock::time_point time_stamp_;
    cv::Vec4f state_;
    cv::Size image_size_;
    cv::Rect2f bbox_;
    double latency_ms_;
    int roi_target_w_ = 0;
    int roi_target_h_ = 0;
};
} // namespace dart_vision