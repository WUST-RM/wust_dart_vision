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
        // 状态维度 4： [x, vx, y, vy]
        // 观测维度 2： [x, y]
        kf_ = cv::KalmanFilter(4, 2, 0, CV_32F);

        // 状态转移矩阵 A
        // [1 dt 0  0]
        // [0 1  0  0]
        // [0 0  1 dt]
        // [0 0  0  1]
        kf_.transitionMatrix =
            (cv::Mat_<float>(4, 4) << 1, dt_, 0, 0, 0, 1, 0, 0, 0, 0, 1, dt_, 0, 0, 0, 1);

        // 观测矩阵 H，观测 x,y 对应状态的第 0 与 2 元素
        // [1 0 0 0]
        // [0 0 1 0]
        kf_.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 0, 1, 0);

        // 过程噪声协方差 Q（对角矩阵，位置与速度可以有不同量级）
        kf_.processNoiseCov = cv::Mat::zeros(4, 4, CV_32F);
        kf_.processNoiseCov.at<float>(0, 0) = procNoisePos;
        kf_.processNoiseCov.at<float>(1, 1) = procNoiseVel;
        kf_.processNoiseCov.at<float>(2, 2) = procNoisePos;
        kf_.processNoiseCov.at<float>(3, 3) = procNoiseVel;

        // 观测噪声协方差 R（2x2）
        kf_.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * measNoise;

        // 先验估计误差协方差 P
        kf_.errorCovPost = cv::Mat::eye(4, 4, CV_32F);

        // 初始化后验状态为 0
        kf_.statePost = cv::Mat::zeros(4, 1, CV_32F);
    }

    // 预测一步（返回预测的状态向量 [x,vx,y,vy]^T）
    cv::Mat predict() {
        return kf_.predict();
    }

    // 使用观测值进行校正，meas_x, meas_y 可以设置为 NAN 表示该维度缺失
    // 返回校正后的状态（posterior）
    cv::Mat correct(float meas_x, float meas_y) {
        cv::Mat measurement = cv::Mat::zeros(2, 1, CV_32F);
        bool hasX = std::isfinite(meas_x);
        bool hasY = std::isfinite(meas_y);

        if (!hasX && !hasY) {
            // 无观测：只做 predict（没有 correct）
            return predict();
        }

        // 若某一个分量缺失，可将对应 measurementNoiseCov 设置为非常大，或者这里直接填入当前预测值
        if (!hasX) {
            measurement.at<float>(0) = kf_.statePre.at<float>(0); // 用预测值填充
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

    // 直接用观测 cv::Point2f
    cv::Mat correct(const cv::Point2f& p) {
        return correct(p.x, p.y);
    }

    // 设置/更新过程噪声（位置和速度）
    void setProcessNoise(float procNoisePos, float procNoiseVel) {
        kf_.processNoiseCov.at<float>(0, 0) = procNoisePos;
        kf_.processNoiseCov.at<float>(1, 1) = procNoiseVel;
        kf_.processNoiseCov.at<float>(2, 2) = procNoisePos;
        kf_.processNoiseCov.at<float>(3, 3) = procNoiseVel;
    }

    // 设置测量噪声（标量，R = measNoise * I）
    void setMeasurementNoise(float measNoise) {
        kf_.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * measNoise;
    }

    // 设置初始状态（x, vx, y, vy）
    void setInitialState(float x, float vx, float y, float vy, float p0 = 1.0f) {
        kf_.statePost.at<float>(0) = x;
        kf_.statePost.at<float>(1) = vx;
        kf_.statePost.at<float>(2) = y;
        kf_.statePost.at<float>(3) = vy;
        kf_.errorCovPost = cv::Mat::eye(4, 4, CV_32F) * p0;
    }

    // 重置滤波器（状态置 0）
    void reset() {
        kf_.statePost = cv::Mat::zeros(4, 1, CV_32F);
        kf_.errorCovPost = cv::Mat::eye(4, 4, CV_32F);
    }

    // 取得当前后验状态 [x,vx,y,vy]
    cv::Vec4f state() const {
        const cv::Mat& s = kf_.statePost;
        return cv::Vec4f(s.at<float>(0), s.at<float>(1), s.at<float>(2), s.at<float>(3));
    }

    // 取得当前预测状态（predict后的 statePre）
    cv::Vec4f predictedState() const {
        const cv::Mat& s = kf_.statePre;
        return cv::Vec4f(s.at<float>(0), s.at<float>(1), s.at<float>(2), s.at<float>(3));
    }

    // 可修改时间步长 dt（并更新 transitionMatrix）
    void setDt(float dt) {
        dt_ = dt;
        kf_.transitionMatrix =
            (cv::Mat_<float>(4, 4) << 1, dt_, 0, 0, 0, 1, 0, 0, 0, 0, 1, dt_, 0, 0, 0, 1);
    }

private:
    cv::KalmanFilter kf_;
    float dt_;
};
} // namespace dart_vision