#pragma once
#include "common.hpp"
#include "kf.hpp"
namespace dart_vision {
class Tracker {
public:
    enum State {
        LOST,
        DETECTING,
        TRACKING,
        TEMP_LOST,
    } tracker_state = LOST;
    Tracker(const Params& params) {
        kf_ = std::make_unique<CVKalman4x2>(
            0.06,
            params.proc_noise_pos,
            params.proc_noise_vel,
            params.meas_noise
        );
        tracking_thres_ = params.tracking_thres;
        lost_dt_ = params.lost_dt;
        max_region_err_ = params.max_region_err;
    }

    void track(const Lights& lights) {
        float dt = std::chrono::duration<float>(lights.timestamp - last_time_).count();
        auto now = std::chrono::steady_clock::now();
        latency_ms_ =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - lights.timestamp).count();
        last_time_ = lights.timestamp;
        lost_thres_ = std::abs(static_cast<int>(lost_dt_ / dt));
        bool found;
        image_size_ = lights.image_size;
        if (tracker_state == LOST) {
            found = init(lights);
        } else {
            found = update(lights);
        }
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
    bool init(const Lights& lights) {
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
        std::cout << "init " << std::endl;
        return true;
    }
    cv::Vec4f predict_future(std::chrono::steady_clock::time_point cur_time) {
        float dt = std::chrono::duration<float>(cur_time - time_stamp_).count();

        return predict_future(dt);
    }
    cv::Vec4f predict_future(float dt) {
        cv::Vec4f state = state_;
        state[0] += dt * state[2];
        state[1] += dt * state[3];
        return state;
    }
    bool update(const Lights& lights) {
        float dt = std::chrono::duration<float>(lights.timestamp - time_stamp_).count();
        kf_->setDt(dt);
        state_ = kf_->predict();
        int best_id = -1;
        float min_error = 1e9;
        for (int i = 0; i < lights.lights.size(); i++) {
            float centor_error =
                cv::norm(lights.lights[i].center - cv::Point2f(state_[0], state_[2]));
            if (centor_error < min_error && centor_error < image_size_.width / max_region_err_) {
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
    void draw(cv::Mat& img) {
        if (tracker_state == TRACKING || tracker_state == TEMP_LOST) {
            cv::Point2f pos(state_(0), state_(2));
            cv::circle(img, pos, 10, cv::Scalar(0, 255, 0), 2);

            float vx = state_(1);
            float vy = state_(3);
            float scale = 1.0f;

            cv::Point2f endPoint(pos.x + vx * scale, pos.y + vy * scale);
            cv::arrowedLine(img, pos, endPoint, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.3);
        }

        std::string text = "Delay: " + std::to_string((int)latency_ms_) + " ms";

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

    bool check() {
        return tracker_state == TRACKING
            || tracker_state == TEMP_LOST
            && std::chrono::duration<float>(std::chrono::steady_clock::now() - time_stamp_).count()
                < 3.0;
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
    cv::Size2f image_size_;
    cv::Rect2f bbox_;
    double latency_ms_;
};
} // namespace dart_vision