#include "light_detector.hpp"
#include "save.hpp"
#include "serial_driver.hpp"
#include "tracker.hpp"
#include "type.hpp"
#include "utils.hpp"
#include "uvc.hpp"
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <queue>
#include <string>
#include <thread>
#include <utility>
using namespace dart_vision;
static std::atomic<bool> stopFlag = false;
bool debug = false;
void signalHandler(int signum) {
    std::cout << "\nSIGINT (Ctrl+C) detected. Stopping safely..." << std::endl;
    stopFlag = true;
}
void registerSignal() {
    struct sigaction sa {};
    sa.sa_handler = signalHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
}
class vision {
public:
    vision(std::string config_file = "config.toml") {
        if (config_file.empty()) {
            config_file = "config.toml";
        }
        const auto config = toml::parse_file(config_file);
        const auto uvc_table = config["uvc"].as_table();
        uvc_ = UVC::create(*uvc_table);
        const auto light_detector_table = config["light_detector"].as_table();
        light_detector_ = LightDetector::create(*light_detector_table);

        const auto camera_info_table = config["camera_info"].as_table();
        std::vector<double> camera_k;
        std::vector<double> camera_d;

        if (camera_info_table->contains("camera_matrix")) {
            if (auto arr = camera_info_table->at("camera_matrix").as_array()) {
                camera_k.reserve(arr->size());
                for (auto& v: *arr) {
                    camera_k.push_back(v.value<double>().value_or(0.0));
                }
            }
        }

        if (camera_info_table->contains("distortion_coefficients")) {
            if (auto arr = camera_info_table->at("distortion_coefficients").as_array()) {
                camera_d.reserve(arr->size());
                for (auto& v: *arr) {
                    camera_d.push_back(v.value<double>().value_or(0.0));
                }
            }
        }

        assert(camera_k.size() == 9);
        assert(camera_d.size() == 5);

        cv::Mat K(3, 3, CV_64F);
        std::memcpy(K.data, camera_k.data(), 9 * sizeof(double));

        cv::Mat D(1, 5, CV_64F);
        std::memcpy(D.data, camera_d.data(), 5 * sizeof(double));

        camera_info_ = std::make_pair(K.clone(), D.clone());
        const auto tracker_table = config["tracker"].as_table();
        tracker_ = Tracker::create(*tracker_table, camera_info_);
        const auto serial_driver_config = config["serial_driver"];
        bool use_serial = serial_driver_config["use_serial"].value_or(false);
        if (use_serial) {
            serial_driver_ = SerialDriver::create();
            const SerialDriver::SerialPortConfig cfg { /*baud*/ 115200,
                                                       /*csize*/ 8,
                                                       asio::serial_port_base::parity::none,
                                                       asio::serial_port_base::stop_bits::one,
                                                       asio::serial_port_base::flow_control::none };
            auto serial_device = serial_driver_config["device"].value_or("");
            serial_driver_->init_port(serial_device, cfg);
            serial_driver_->set_receive_callback(std::bind(
                &vision::serialCallback,
                this,
                std::placeholders::_1,
                std::placeholders::_2
            ));
            std::cout << "Serial port opened" << std::endl;
            serial_driver_->set_error_callback([&](const asio::error_code& ec) {
                std::cout << "serial error: " << ec.message() << std::endl;
                // serial_driver_->close_port();
                // serial_driver_->open_port();
                //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            });
            serial_driver_->open_port();
        }
        const auto video_saver_config = config["video_saver"];
        bool save_video = video_saver_config["enable"].value_or(false);
        if (save_video) {
            video_saver_ =
                std::make_unique<dart_vision::VideoSaver>(getTimestampFilename(), WriteMode::ASYNC);
            video_saver_->start();
        }
    }
    void serialCallback(const uint8_t* data, std::size_t len) noexcept {}
    void run() {
        capture_thread_ = std::thread(&vision::captureThread, this);
        process_thread_ = std::thread(&vision::processThread, this);
        control_thread_ = std::thread(&vision::controlThread, this);
        capture_thread_.join();
        process_thread_.join();
        control_thread_.join();
    }
    void printStats() noexcept {
        XSecOnce(
            [&]() {
                std::cout << "cap hz:" << capture_count_ << " cost: " << capture_cost_ms_ << " ms"
                          << " pro hz:" << process_count_ << " cost: " << process_cost_ms_ << " ms"
                          << " control hz:" << control_count_ << std::endl;
                capture_cost_ms_ = 0;
                process_cost_ms_ = 0;
                process_count_ = 0;
                capture_count_ = 0;
                control_count_ = 0;
            },
            1.0
        );
    }
    void processAFrame(Frame& frame) noexcept {
        bool need_show = false;
        bool need_draw = false;
        if (debug) {
            need_show = true;
            need_draw = true;
        } else if (video_saver_) {
            need_draw = true;
        }
        process_count_++;
        const auto start = std::chrono::steady_clock::now();
        auto& image = frame.image;
        frame.expanded = cv::Rect(0, 0, image.cols, image.rows);
        frame.offset = cv::Point2f(0, 0);
        if (tracker_->check()) {
            const auto track_bbox = tracker_->expanded();
            frame.expanded = track_bbox;
            frame.offset = track_bbox.tl();
        }
        const auto lights = light_detector_->detect(frame, need_show, need_draw);

        tracker_->track(lights);

        if (need_draw) {
            cv::rectangle(image, frame.expanded, cv::Scalar(0, 255, 0), 2);
            tracker_->draw(image);
        }

        const auto end = std::chrono::steady_clock::now();
        process_cost_ms_ +=
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printStats();
        if (need_show) {
            writeFrameToShm<10>(image);
        }
#ifdef X86
        if (need_show) {
            cv::imshow("image", image);
            cv::waitKey(1);
        }
#endif
    }
    void processThread() noexcept {
        Frame frame;
        while (!stopFlag) {
            {
                std::unique_lock<std::mutex> lock(frame_queue_mutex_);
                frame_queue_condition_.wait(lock, [this] {
                    return !frame_queue_.empty() || stopFlag;
                });
                if (!frame_queue_.empty()) {
                    frame = std::move(frame_queue_.front());
                    frame_queue_.pop();
                }
            }
            processAFrame(frame);
        }
    }
    void captureThread() noexcept {
        Frame frame;
        cv::Mat image;
        while (!stopFlag) {
            capture_count_++;
            auto start = std::chrono::steady_clock::now();
            image = uvc_->read(10);
            if (image.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            frame.image = std::move(image);
            frame.timestamp = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lock(frame_queue_mutex_);
                frame_queue_.push(std::move(frame));
            }
            frame_queue_condition_.notify_one();
            auto end = std::chrono::steady_clock::now();
            capture_cost_ms_ +=
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }
    }
    void controlThread() noexcept {
        auto next_time = std::chrono::steady_clock::now();
        while (!stopFlag) {
            control_count_++;
            const auto now = std::chrono::steady_clock::now();
            cv::Point2f p;
            SerialFrame f;
            if (tracker_->check()) {
                auto state = tracker_->predict_future(now);
                auto x = state.first[0];
                auto y = state.first[2];
                auto dis = state.second[0];
                std::vector<cv::Point2f> src = { cv::Point2f(x, y) };
                std::vector<cv::Point2f> dst;

                cv::undistortPoints(src, dst, camera_info_.first, camera_info_.second);

                float xn = dst[0].x;
                float yn = dst[0].y;

                float yaw = std::atan(xn) ;
                float pitch = std::atan(-yn) ;
                
                f.pitch = -pitch;
                f.yaw = -yaw;
                f.dis = dis;
            } else {
                f.pitch = 0;
                f.yaw = 0;
                f.dis = 0;
            }

            uint32_t sum = 0;
            uint8_t* pData = reinterpret_cast<uint8_t*>(&f);
            for (int i = 0; i < sizeof(SerialFrame) - sizeof(f.sum); i++) {
                sum += pData[i];
            }

            f.sum = sum;
            if (!serial_driver_->write(toVector(f))) {
                serial_driver_->close_port();
                serial_driver_->open_port();
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }

            next_time += std::chrono::milliseconds(1);
            std::this_thread::sleep_until(next_time);
        }
    }

private:
    std::thread capture_thread_;
    std::thread process_thread_;
    std::thread control_thread_;
    UVC::Ptr uvc_;
    LightDetector::Ptr light_detector_;
    Tracker::Ptr tracker_;
    SerialDriver::Ptr serial_driver_;
    VideoSaver::Ptr video_saver_;
    std::queue<Frame> frame_queue_;
    mutable std::mutex frame_queue_mutex_;
    std::condition_variable frame_queue_condition_;
    std::pair<cv::Mat, cv::Mat> camera_info_;
    int capture_count_ = 0;
    int process_count_ = 0;
    int control_count_ = 0;
    double capture_cost_ms_ = 0;
    double process_cost_ms_ = 0;
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