// kf_cv.h
#pragma once
#include <opencv2/opencv.hpp>
namespace dart_vision {
class CVKalman4x2 {
public:
    CVKalman4x2(
        float dt = 1.0f,
        float procNoisePos = 1e-2f,
        float procNoiseVel = 1e-3f,
        float measNoise = 1.0f
    ):
        dt_(dt) {
        kf_ = cv::KalmanFilter(4, 2, 0, CV_32F);

        kf_.transitionMatrix =
            (cv::Mat_<float>(4, 4) << 1, dt_, 0, 0, 0, 1, 0, 0, 0, 0, 1, dt_, 0, 0, 0, 1);
        kf_.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 0, 1, 0);

        kf_.processNoiseCov = cv::Mat::zeros(4, 4, CV_32F);
        kf_.processNoiseCov.at<float>(0, 0) = procNoisePos;
        kf_.processNoiseCov.at<float>(1, 1) = procNoiseVel;
        kf_.processNoiseCov.at<float>(2, 2) = procNoisePos;
        kf_.processNoiseCov.at<float>(3, 3) = procNoiseVel;

        kf_.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * measNoise;

        kf_.errorCovPost = cv::Mat::eye(4, 4, CV_32F);

        kf_.statePost = cv::Mat::zeros(4, 1, CV_32F);
    }

    cv::Mat predict() noexcept {
        return kf_.predict();
    }

    cv::Mat correct(float meas_x, float meas_y) noexcept {
        cv::Mat measurement = cv::Mat::zeros(2, 1, CV_32F);
        bool hasX = std::isfinite(meas_x);
        bool hasY = std::isfinite(meas_y);

        if (!hasX && !hasY) {
            return predict();
        }

        if (!hasX) {
            measurement.at<float>(0) = kf_.statePre.at<float>(0);
        } else {
            measurement.at<float>(0) = meas_x;
        }
        if (!hasY) {
            measurement.at<float>(1) = kf_.statePre.at<float>(2);
        } else {
            measurement.at<float>(1) = meas_y;
        }

        return kf_.correct(measurement);
    }

    cv::Mat correct(const cv::Point2f& p) noexcept {
        return correct(p.x, p.y);
    }

    void setProcessNoise(float procNoisePos, float procNoiseVel) noexcept {
        kf_.processNoiseCov.at<float>(0, 0) = procNoisePos;
        kf_.processNoiseCov.at<float>(1, 1) = procNoiseVel;
        kf_.processNoiseCov.at<float>(2, 2) = procNoisePos;
        kf_.processNoiseCov.at<float>(3, 3) = procNoiseVel;
    }

    void setMeasurementNoise(float measNoise) noexcept {
        kf_.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * measNoise;
    }

    void setInitialState(float x, float vx, float y, float vy, float p0 = 1.0f) noexcept {
        kf_.statePost.at<float>(0) = x;
        kf_.statePost.at<float>(1) = vx;
        kf_.statePost.at<float>(2) = y;
        kf_.statePost.at<float>(3) = vy;
        kf_.errorCovPost = cv::Mat::eye(4, 4, CV_32F) * p0;
    }

    void reset() noexcept {
        kf_.statePost = cv::Mat::zeros(4, 1, CV_32F);
        kf_.errorCovPost = cv::Mat::eye(4, 4, CV_32F);
    }

    cv::Vec4f state() const noexcept {
        const cv::Mat& s = kf_.statePost;
        return cv::Vec4f(s.at<float>(0), s.at<float>(1), s.at<float>(2), s.at<float>(3));
    }
    cv::Vec4f predictedState() const noexcept {
        const cv::Mat& s = kf_.statePre;
        return cv::Vec4f(s.at<float>(0), s.at<float>(1), s.at<float>(2), s.at<float>(3));
    }

    void setDt(float dt) noexcept {
        dt_ = dt;
        kf_.transitionMatrix =
            (cv::Mat_<float>(4, 4) << 1, dt_, 0, 0, 0, 1, 0, 0, 0, 0, 1, dt_, 0, 0, 0, 1);
    }

private:
    cv::KalmanFilter kf_;
    float dt_;
};
class CVKalman2x1 {
public:
    CVKalman2x1(
        float dt = 1.0f,
        float procNoisePos = 1e-2f,
        float procNoiseVel = 1e-3f,
        float measNoise = 1.0f
    ):
        dt_(dt) {
        kf_ = cv::KalmanFilter(2, 1, 0, CV_32F);

        // 状态转移矩阵
        // [x]
        // [vx]
        kf_.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, dt_, 0, 1);

        // 观测矩阵
        // 只观测位置
        kf_.measurementMatrix = (cv::Mat_<float>(1, 2) << 1, 0);

        // 过程噪声
        kf_.processNoiseCov = cv::Mat::zeros(2, 2, CV_32F);
        kf_.processNoiseCov.at<float>(0, 0) = procNoisePos;
        kf_.processNoiseCov.at<float>(1, 1) = procNoiseVel;

        // 测量噪声
        kf_.measurementNoiseCov = cv::Mat::eye(1, 1, CV_32F) * measNoise;

        // 误差协方差
        kf_.errorCovPost = cv::Mat::eye(2, 2, CV_32F);

        // 初始状态
        kf_.statePost = cv::Mat::zeros(2, 1, CV_32F);
    }

    cv::Mat predict() noexcept {
        return kf_.predict();
    }

    cv::Mat correct(float meas_x) noexcept {
        if (!std::isfinite(meas_x))
            return predict();

        cv::Mat measurement(1, 1, CV_32F);
        measurement.at<float>(0) = meas_x;

        return kf_.correct(measurement);
    }

    void setProcessNoise(float procNoisePos, float procNoiseVel) noexcept {
        kf_.processNoiseCov.at<float>(0, 0) = procNoisePos;
        kf_.processNoiseCov.at<float>(1, 1) = procNoiseVel;
    }

    void setMeasurementNoise(float measNoise) noexcept {
        kf_.measurementNoiseCov = cv::Mat::eye(1, 1, CV_32F) * measNoise;
    }

    void setInitialState(float x, float vx, float p0 = 1.0f) noexcept {
        kf_.statePost.at<float>(0) = x;
        kf_.statePost.at<float>(1) = vx;
        kf_.errorCovPost = cv::Mat::eye(2, 2, CV_32F) * p0;
    }

    void reset() noexcept {
        kf_.statePost = cv::Mat::zeros(2, 1, CV_32F);
        kf_.errorCovPost = cv::Mat::eye(2, 2, CV_32F);
    }

    cv::Vec2f state() const noexcept {
        const cv::Mat& s = kf_.statePost;
        return cv::Vec2f(
            s.at<float>(0), // x
            s.at<float>(1) // vx
        );
    }

    cv::Vec2f predictedState() const noexcept {
        const cv::Mat& s = kf_.statePre;
        return cv::Vec2f(s.at<float>(0), s.at<float>(1));
    }

    void setDt(float dt) noexcept {
        dt_ = dt;

        kf_.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, dt_, 0, 1);
    }

private:
    cv::KalmanFilter kf_;
    float dt_;
};
class CVKalman6x3 {
public:
    CVKalman6x3(
        float dt = 1.0f,
        float procNoisePos = 1e-2f,
        float procNoiseVel = 1e-3f,
        float measNoise = 1.0f
    ):
        dt_(dt) {
        kf_ = cv::KalmanFilter(6, 3, 0, CV_32F);

        // 状态: [x vx y vy z vz]
        kf_.transitionMatrix =
            (cv::Mat_<float>(6, 6) << 1,
             dt_,
             0,
             0,
             0,
             0,
             0,
             1,
             0,
             0,
             0,
             0,
             0,
             0,
             1,
             dt_,
             0,
             0,
             0,
             0,
             0,
             1,
             0,
             0,
             0,
             0,
             0,
             0,
             1,
             dt_,
             0,
             0,
             0,
             0,
             0,
             1);

        // 观测矩阵: 只观测位置 [x y z]
        kf_.measurementMatrix =
            (cv::Mat_<float>(3, 6) << 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0);

        // 过程噪声
        kf_.processNoiseCov = cv::Mat::zeros(6, 6, CV_32F);
        kf_.processNoiseCov.at<float>(0, 0) = procNoisePos;
        kf_.processNoiseCov.at<float>(1, 1) = procNoiseVel;
        kf_.processNoiseCov.at<float>(2, 2) = procNoisePos;
        kf_.processNoiseCov.at<float>(3, 3) = procNoiseVel;
        kf_.processNoiseCov.at<float>(4, 4) = procNoisePos;
        kf_.processNoiseCov.at<float>(5, 5) = procNoiseVel;

        // 测量噪声
        kf_.measurementNoiseCov = cv::Mat::eye(3, 3, CV_32F) * measNoise;

        // 协方差
        kf_.errorCovPost = cv::Mat::eye(6, 6, CV_32F);

        // 初始状态
        kf_.statePost = cv::Mat::zeros(6, 1, CV_32F);
    }

    cv::Mat predict() noexcept {
        return kf_.predict();
    }

    cv::Mat correct(float meas_x, float meas_y, float meas_z) noexcept {
        bool hasX = std::isfinite(meas_x);
        bool hasY = std::isfinite(meas_y);
        bool hasZ = std::isfinite(meas_z);

        if (!hasX && !hasY && !hasZ)
            return predict();

        cv::Mat measurement = cv::Mat::zeros(3, 1, CV_32F);

        measurement.at<float>(0) = hasX ? meas_x : kf_.statePre.at<float>(0);
        measurement.at<float>(1) = hasY ? meas_y : kf_.statePre.at<float>(2);
        measurement.at<float>(2) = hasZ ? meas_z : kf_.statePre.at<float>(4);

        return kf_.correct(measurement);
    }

    cv::Mat correct(const cv::Point3f& p) noexcept {
        return correct(p.x, p.y, p.z);
    }

    void setProcessNoise(float pos, float vel) noexcept {
        kf_.processNoiseCov.at<float>(0, 0) = pos;
        kf_.processNoiseCov.at<float>(1, 1) = vel;
        kf_.processNoiseCov.at<float>(2, 2) = pos;
        kf_.processNoiseCov.at<float>(3, 3) = vel;
        kf_.processNoiseCov.at<float>(4, 4) = pos;
        kf_.processNoiseCov.at<float>(5, 5) = vel;
    }

    void setMeasurementNoise(float measNoise) noexcept {
        kf_.measurementNoiseCov = cv::Mat::eye(3, 3, CV_32F) * measNoise;
    }

    void setInitialState(
        float x,
        float vx,
        float y,
        float vy,
        float z,
        float vz,
        float p0 = 1.0f
    ) noexcept {
        kf_.statePost.at<float>(0) = x;
        kf_.statePost.at<float>(1) = vx;
        kf_.statePost.at<float>(2) = y;
        kf_.statePost.at<float>(3) = vy;
        kf_.statePost.at<float>(4) = z;
        kf_.statePost.at<float>(5) = vz;

        kf_.errorCovPost = cv::Mat::eye(6, 6, CV_32F) * p0;
    }

    void reset() noexcept {
        kf_.statePost = cv::Mat::zeros(6, 1, CV_32F);
        kf_.errorCovPost = cv::Mat::eye(6, 6, CV_32F);
    }

    cv::Vec6f state() const noexcept {
        const cv::Mat& s = kf_.statePost;
        return cv::Vec6f(
            s.at<float>(0),
            s.at<float>(1),
            s.at<float>(2),
            s.at<float>(3),
            s.at<float>(4),
            s.at<float>(5)
        );
    }

    cv::Vec6f predictedState() const noexcept {
        const cv::Mat& s = kf_.statePre;
        return cv::Vec6f(
            s.at<float>(0),
            s.at<float>(1),
            s.at<float>(2),
            s.at<float>(3),
            s.at<float>(4),
            s.at<float>(5)
        );
    }

    void setDt(float dt) noexcept {
        dt_ = dt;
        kf_.transitionMatrix =
            (cv::Mat_<float>(6, 6) << 1,
             dt_,
             0,
             0,
             0,
             0,
             0,
             1,
             0,
             0,
             0,
             0,
             0,
             0,
             1,
             dt_,
             0,
             0,
             0,
             0,
             0,
             1,
             0,
             0,
             0,
             0,
             0,
             0,
             1,
             dt_,
             0,
             0,
             0,
             0,
             0,
             1);
    }

private:
    cv::KalmanFilter kf_;
    float dt_;
};
} // namespace dart_vision