#include "light_detector.hpp"
#include "save.hpp"
#include "serial_driver.hpp"
#include "tracker.hpp"
#include "uvc.hpp"
#include <condition_variable>
#include <csignal>
#include <queue>
#include <thread>
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
    vision() {
        auto config = toml::parse_file("../config.toml");
        auto uvc_table = config["uvc"].as_table();
        uvc_ = UVC::create(*uvc_table);
        auto light_detector_table = config["light_detector"].as_table();
        light_detector_ = LightDetector::create(*light_detector_table);
        auto tracker_table = config["tracker"].as_table();
        tracker_ = Tracker::create(*tracker_table);
        auto serial_driver_config = config["serial_driver"];
        bool use_serial = serial_driver_config["use_serial"].value_or(false);
        if (use_serial) {
            serial_driver_ = SerialDriver::create();
            SerialDriver::SerialPortConfig cfg { /*baud*/ 115200,
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
            });
            serial_driver_->start();
        }
        auto video_saver_config = config["video_saver"];
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
        auto start = std::chrono::steady_clock::now();
        auto& image = frame.image;
        frame.expanded = cv::Rect(0, 0, image.cols, image.rows);
        frame.offset = cv::Point2f(0, 0);
        if (tracker_->check()) {
            const auto track_bbox = tracker_->expanded();
            frame.expanded = track_bbox;
            frame.offset = track_bbox.tl();
        }
        auto lights = light_detector_->detect(frame, need_show, need_draw);
        tracker_->track(lights);
        if (need_draw) {
            cv::rectangle(image, frame.expanded, cv::Scalar(0, 255, 0), 2);
            tracker_->draw(image);
        }

        auto end = std::chrono::steady_clock::now();
        process_cost_ms_ +=
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printStats();
        if (need_show) {
            writeFrameToShm(image);
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
            if (!uvc_->cap().grab()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            capture_count_++;
            auto start = std::chrono::steady_clock::now();
            uvc_->cap().retrieve(image);
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
            auto now = std::chrono::steady_clock::now();
            cv::Point2f p;
            if (tracker_->check()) {
                auto state = tracker_->predict_future(now);
                p.x = state[0];
                p.y = state[2];
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
    std::mutex frame_queue_mutex_;
    std::condition_variable frame_queue_condition_;
    int capture_count_ = 0;
    int process_count_ = 0;
    int control_count_ = 0;
    double capture_cost_ms_ = 0;
    double process_cost_ms_ = 0;
};

int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);
    if (argc > 1) {
        std::string firstArg = argv[1];
        debug = (firstArg == "true" || firstArg == "1");
        std::cout << "debug: " << debug << std::endl;
    }
    vision v;
    v.run();

    return 0;
}