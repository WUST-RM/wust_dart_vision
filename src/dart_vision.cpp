#include "light_detector.hpp"
#include "save.hpp"
#include "serial_driver.hpp"
#include "tracker.hpp"
#include "type.hpp"
#include "utils.hpp"
#include "uvc.hpp"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

using namespace dart_vision;

static std::atomic<bool> stopFlag{false};
static bool debug = false;

void signalHandler(int /*signum*/) {
    stopFlag.store(true, std::memory_order_relaxed);
}

class vision {
public:
    explicit vision(const std::string& config_file = "config.toml") {
        // parse config (kept same semantics)
        const auto config = toml::parse_file(config_file);
        uvc_ = UVC::create(*config["uvc"].as_table());
        light_detector_ = LightDetector::create(*config["light_detector"].as_table());

        // camera intrinsics
        const auto camera_info_table = config["camera_info"].as_table();
        std::vector<double> camera_k;
        std::vector<double> camera_d;
        if (auto arr = camera_info_table->at("camera_matrix").as_array()) {
            camera_k.reserve(arr->size());
            for (auto &v : *arr) camera_k.push_back(v.value<double>().value_or(0.0));
        }
        if (auto arr = camera_info_table->at("distortion_coefficients").as_array()) {
            camera_d.reserve(arr->size());
            for (auto &v : *arr) camera_d.push_back(v.value<double>().value_or(0.0));
        }

        if (camera_k.size() != 9 || camera_d.size() != 5) {
            throw std::runtime_error("camera_matrix/distortion_coefficients size mismatch");
        }

        // create mats with correct types and avoid memcpy of float->double
        cv::Mat K(3, 3, CV_64F);
        for (int i = 0; i < 9; ++i) K.at<double>(i / 3, i % 3) = camera_k[i];
        cv::Mat D(1, 5, CV_64F);
        for (int i = 0; i < 5; ++i) D.at<double>(0, i) = camera_d[i];

        camera_info_ = {K.clone(), D.clone()};

        tracker_ = Tracker::create(*config["tracker"].as_table(), camera_info_);

        // serial init (unchanged semantics)
        if (config["serial_driver"]["use_serial"].value_or(false)) {
            serial_driver_ = SerialDriver::create();
            SerialDriver::SerialPortConfig cfg {115200, 8,
                asio::serial_port_base::parity::none,
                asio::serial_port_base::stop_bits::one,
                asio::serial_port_base::flow_control::none};
            auto dev = config["serial_driver"]["device"].value_or("");
            serial_driver_->init_port(dev, cfg);
            serial_driver_->set_receive_callback([this](const uint8_t* data, std::size_t len) {
                this->serialCallback(data, len);
            });
            serial_driver_->start();
        }

        if (config["video_saver"]["enable"].value_or(false)) {
            video_saver_ = std::make_unique<VideoSaver>(getTimestampFilename(), WriteMode::ASYNC);
            video_saver_->start();
        }

        // precompute constant camera->gimbal matrix (does not depend on frame)
        R_camera2gimbal_ << 0.0f, 0.0f, 1.0f,
                           -1.0f, -0.0f, 0.0f,
                            0.0f, -1.0f, 0.0f;

        // params
        max_queue_size_ = 4; // keep small to avoid latency buildup; tune for your pipeline
    }

    void run() {
        capture_thread_ = std::thread(&vision::captureThread, this);
        process_thread_ = std::thread(&vision::processThread, this);
        control_thread_ = std::thread(&vision::controlThread, this);

        capture_thread_.join();
        process_thread_.join();
        control_thread_.join();
    }

private:
    // ----- callbacks / threads -----
    void serialCallback(const uint8_t* data, std::size_t len) noexcept {
        // parse in-place to avoid extra vector copy where possible
        try {
            auto frame = fromVector<ReceiveFrame>(std::vector<uint8_t>(data, data + len));
            last_receive_frame_ = frame;
        } catch (...) {
            // ignore malformed
        }
    }

    void captureThread() noexcept {
        while (!stopFlag.load(std::memory_order_relaxed)) {
            auto start = std::chrono::steady_clock::now();
            cv::Mat image = uvc_->read(10);
            if (image.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            FramePtr f = std::make_unique<Frame>();
            f->image = std::move(image);
            f->timestamp = std::chrono::steady_clock::now();

            {
                std::lock_guard<std::mutex> lk(queue_mutex_);
                // bounded queue: drop oldest frame when full (prefer newest)
                if (frame_queue_.size() >= max_queue_size_) {
                    frame_queue_.pop_front();
                }
                frame_queue_.push_back(std::move(f));
            }
            queue_cv_.notify_one();

            auto end = std::chrono::steady_clock::now();
            ++capture_count_;
            capture_cost_ms_ += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }
    }

    void processThread() noexcept {
        while (!stopFlag.load(std::memory_order_relaxed)) {
            FramePtr frame;
            {
                std::unique_lock<std::mutex> lk(queue_mutex_);
                queue_cv_.wait(lk, [this] {
                    return !frame_queue_.empty() || stopFlag.load(std::memory_order_relaxed);
                });
                if (frame_queue_.empty()) continue;
                frame = std::move(frame_queue_.front());
                frame_queue_.pop_front();
            }
            if (frame) processAFrame(*frame);
        }
    }

    void controlThread() noexcept {
        auto next_time = std::chrono::steady_clock::now();
        while (!stopFlag.load(std::memory_order_relaxed)) {
            ++control_count_;
            SendFrame f{};
            if (tracker_->check()) {
                auto state = tracker_->predict_future(std::chrono::steady_clock::now());
                float dx = static_cast<float>(state[0]);
                float dy = static_cast<float>(state[2]);
                float dz = static_cast<float>(state[4]);
                float dis = std::sqrt(dx*dx + dy*dy + dz*dz);
                float yaw = std::atan2(dy, dx);
                float pitch = std::asin(dz / std::max(dis, 1e-6f));
                f.pitch = -pitch;
                f.yaw = -yaw;
                f.dis = dis;
            }

            // checksum
            uint32_t sum = 0;
            uint8_t* pData = reinterpret_cast<uint8_t*>(&f);
            for (size_t i = 0; i < sizeof(SendFrame) - sizeof(f.sum); ++i) sum += pData[i];
            f.sum = sum;

            if (serial_driver_) {
                if (!serial_driver_->write(toVector(f))) {
                    // avoid blocking here; schedule reconnect asynchronously would be better
                    serial_driver_->close_port();
                    serial_driver_->open_port();
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }

            next_time += std::chrono::milliseconds(1);
            std::this_thread::sleep_until(next_time);
        }
    }

    // ----- processing -----
    void processAFrame(Frame& frame) noexcept {
        const bool need_show = debug;
        const bool need_draw = debug || static_cast<bool>(video_saver_);

        ++process_count_;
        auto start = std::chrono::steady_clock::now();

        auto& image = frame.image;
        frame.expanded = cv::Rect(0, 0, image.cols, image.rows);
        frame.offset = {0, 0};

        if (tracker_->check()) {
            auto track_bbox = tracker_->expanded();
            frame.expanded = track_bbox;
            frame.offset = track_bbox.tl();
        }

        const auto lights = light_detector_->detect(frame, need_show, need_draw);
        Eigen::Matrix3f R_gimbal2odom = Eigen::Matrix3f::Identity();
        R_gimbal2odom = Eigen::AngleAxisf(0.f, Eigen::Vector3f::UnitZ())
            * Eigen::AngleAxisf(last_receive_frame_.pitch, Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf(last_receive_frame_.roll, Eigen::Vector3f::UnitX());

        static Eigen::Matrix4f T_gimbal_to_odom = Eigen::Matrix4f::Identity();
        T_gimbal_to_odom.block<3,3>(0,0) = R_gimbal2odom;

        static Eigen::Matrix4f T_camera_to_gimbal = Eigen::Matrix4f::Identity();
        T_camera_to_gimbal.block<3,3>(0,0) = R_camera2gimbal_;
        T_camera_to_gimbal.block<3,1>(0,3) = Eigen::Vector3f::Zero();

        T_camera_to_odom_ = T_gimbal_to_odom * T_camera_to_gimbal;
        tracker_->track(lights, T_camera_to_odom_);

        if (need_draw) {
            cv::rectangle(image, frame.expanded, cv::Scalar(0,255,0), 2);
            tracker_->draw(image);
        }

        if (need_show) {
            writeFrameToShm<10>(image);
#ifdef X86
            if (need_show) {
                cv::imshow("image", image);
                cv::waitKey(1);
            }
#endif
        }

        auto end = std::chrono::steady_clock::now();
        process_cost_ms_ += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    // ----- members -----
    std::thread capture_thread_, process_thread_, control_thread_;
    UVC::Ptr uvc_;
    LightDetector::Ptr light_detector_;
    Tracker::Ptr tracker_;
    SerialDriver::Ptr serial_driver_;
    std::unique_ptr<VideoSaver> video_saver_;

    std::deque<FramePtr> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    size_t max_queue_size_{4};

    std::pair<cv::Mat, cv::Mat> camera_info_;

    int capture_count_{0}, process_count_{0}, control_count_{0};
    double capture_cost_ms_{0}, process_cost_ms_{0};
    Eigen::Matrix4f T_camera_to_odom_ = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f R_camera2gimbal_;
    ReceiveFrame last_receive_frame_;
};

int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);
    std::string config_file;
    if (argc > 2) {
        config_file = argv[1];
        std::cout << "config_file: " << config_file << std::endl;
        std::string firstArg = argv[2];
        debug = (firstArg == "true" || firstArg == "1");
        std::cout << "debug: " << debug << std::endl;
    }
    vision v(config_file);
    v.run();

    return 0;
}