#include "common.hpp"
#include "debug_utils.hpp"
#include "kf.hpp"
#ifdef ARM
    #include "neon.hpp"
#endif
#include "package.hpp"
#include "save.hpp"
#include "serial_driver.hpp"
#include "tracker.hpp"
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#define QUEUE_SIZE 7

static std::queue<dart_vision::Frame> frameQueue;
static std::mutex queueMutex;
static std::condition_variable queueCond;
static std::atomic<bool> stopFlag(false);
bool debug = false;
dart_vision::Params params;
std::unique_ptr<dart_vision::Tracker> tracker;
std::unique_ptr<dart_vision::VideoSaver> saver;
dart_vision::SerialDriver serial;
int capture_frame_count = 0;
int control_frame_count = 0;
int process_frame_count = 0;
bool use_saver = false;
std::string findAvailableCamera(std::string priority_dev) {
    std::cout << "Searching for cameras..." << std::endl;
    cv::VideoCapture cap(priority_dev, cv::CAP_V4L2);
    if (cap.isOpened()) {
        std::cout << "Found camera: " << priority_dev << std::endl;
        cap.release();
        return priority_dev;
    }
    for (int i = 0; i < 10; i++) {
        std::string device = "/dev/video" + std::to_string(i);
        cv::VideoCapture cap(device, cv::CAP_V4L2);
        if (cap.isOpened()) {
            std::cout << "Found camera: " << device << std::endl;
            cap.release();
            return device;
        }
    }
    return "";
}
void processAFrame(const dart_vision::Frame& frame) {
    cv::Mat hsv, mask, image;
    int last_id = -1;
    dart_vision::Lights lights;
    bool tracker_online = (tracker) && tracker->check();
    int scale_percent = std::max(1, params.scale_percent);
    double scale = scale_percent / 100.0;
    if (scale > 99) {
        image = std::move(frame.image);
    } else {
        cv::resize(frame.image, image, cv::Size(), scale, scale, cv::INTER_NEAREST);
    }
    if (image.channels() != 3) {
        std::cerr << "Warning: input frame channels != 3, skipping\n";
        return;
    }
    lights.image_size = image.size();
    cv::Rect expanded = cv::Rect(0, 0, image.cols, image.rows);
    cv::Point2f offset = cv::Point2f(0, 0);
    if (tracker_online) {
        const int TARGET_W = params.roi_target_w;
        const int TARGET_H = params.roi_target_h;
        const float ASPECT = (float)TARGET_W / TARGET_H;
        const int TARGET_AREA = TARGET_W * TARGET_H;

        auto bbox = tracker->bbox_;
        int bw = bbox.width, bh = bbox.height;

        // ------- Step1: adjust aspect ratio -------
        int w = bw, h = bh;
        if ((float)bw / bh > ASPECT)
            h = (int)(bw / ASPECT);
        else
            w = (int)(bh * ASPECT);

        // ------- Step2: scale to target area -------
        float scale = std::sqrt((float)TARGET_AREA / (w * h));
        w = (int)(w * scale);
        h = (int)(h * scale);

        // ensure exact aspect
        if ((float)w / h > ASPECT)
            h = (int)(w / ASPECT);
        else
            w = (int)(h * ASPECT);

        // ------- Step3: center around bbox -------
        int cx = bbox.x + bw / 2;
        int cy = bbox.y + bh / 2;

        int x = cx - w / 2;
        int y = cy - h / 2;

        // ------- Step4: clamp inside image -------
        x = std::clamp(x, 0, image.cols - w);
        y = std::clamp(y, 0, image.rows - h);

        expanded = cv::Rect(x, y, w, h);
        offset = cv::Point2f(x, y);
    }

    cv::Mat roi = image(expanded);

#ifdef X86
    cv::Mat G, B, R;
    cv::extractChannel(roi, G, 1);
    cv::extractChannel(roi, R, 2);
    cv::extractChannel(roi, B, 0);
    if (B.type() != CV_8U) {
        B.convertTo(B, CV_8U);
    }
    if (G.type() != CV_8U) {
        G.convertTo(G, CV_8U);
    }
    if (R.type() != CV_8U) {
        R.convertTo(R, CV_8U);
    }

    cv::Mat maxRB;
    cv::max(R, B, maxRB);

    cv::Mat diff;
    cv::subtract(G, maxRB, diff); // diff = G - max(R,B)
    if (diff.type() != CV_8U) {
        diff.convertTo(diff, CV_8U);
    }

    cv::threshold(diff, mask, params.D_thresh, 255, cv::THRESH_BINARY);
#endif
#ifdef ARM
    mask = cv::Mat(roi.rows, roi.cols, CV_8UC1);
    dart_vision::neon_diff_threshold_dispatch(roi, mask, static_cast<uint8_t>(params.D_thresh));
#endif
    const float ratio_diff = params.one_one_diff;
    lights.timestamp = frame.timestamp;
    lights.lights.clear();

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    lights.lights.clear();
    
    for (int i = 1; i < nLabels; i++) { // i=0 是背景
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        float cx = centroids.at<double>(i, 0);
        float cy = centroids.at<double>(i, 1);

        float ratio = float(w) / float(h);
        if (ratio < 1 - params.one_one_diff || ratio > 1 + params.one_one_diff)
            continue;

        dart_vision::Light light;
        light.area = area;
        light.center = cv::Point2f(cx, cy) + offset;
        light.bbox = cv::Rect2f(x + offset.x, y + offset.y, w, h);
        light.timestamp = frame.timestamp;
        light.valid = true;
        lights.lights.push_back(light);

        if (debug) {
            cv::rectangle(
                image,
                { x + (int)offset.x, y + (int)offset.y, w, h },
                cv::Scalar(255, 0, 0),
                2
            );
            cv::circle(
                image,
                cv::Point2f(cx + offset.x, cy + offset.y),
                3,
                cv::Scalar(0, 255, 0),
                -1
            );
        }
    }
    if (debug) {
        cv::rectangle(image, expanded, cv::Scalar(0, 255, 0), 2); // 绿色边框
    }
    if (tracker)
        tracker->track(lights);
    if (debug || use_saver && tracker)
        tracker->draw(image);

    process_frame_count++;

    if (debug) {
#ifdef ARM
        dart_vision::writeFrameToShm(image);
#endif
#ifdef X86
        cv::imshow("frame", image);
        cv::imshow("mask", mask);
        cv::waitKey(1);
#endif
    }
    if (saver) {
        saver->pushFrame(image);
    }
}
void captureThreadFunc() {
    std::string dev = findAvailableCamera(params.device_name);
    if (dev.empty()) {
        std::cerr << "No camera detected!" << std::endl;
        stopFlag = true;
        queueCond.notify_all();
        return;
    }

    cv::VideoCapture cap(dev, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera: " << dev << std::endl;
        stopFlag = true;
        queueCond.notify_all();
        return;
    }

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FPS, 60);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, params.image_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, params.image_height);
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
    cap.set(cv::CAP_PROP_EXPOSURE, params.exposure);
    cap.set(cv::CAP_PROP_GAIN, 0);

    int current_id = 0;
    while (!stopFlag) {
        cv::Mat _frame;
        if (!cap.grab())
            continue; // 非阻塞
        cap.retrieve(_frame);

        dart_vision::Frame frame;
        frame.id = current_id++;
        frame.image = std::move(_frame);
        frame.timestamp = std::chrono::steady_clock::now();
        // processAFrame(frame);
        {
            std::lock_guard<std::mutex> lk(queueMutex);
            if (frameQueue.size() >= QUEUE_SIZE)
                frameQueue.pop();
            frameQueue.push(std::move(frame));
        }
        queueCond.notify_one();

        capture_frame_count++;
    }
}

void processThreadFunc() {
    while (!stopFlag) {
        dart_vision::Frame frame;
        {
            std::unique_lock<std::mutex> lk(queueMutex);
            queueCond.wait_for(lk, std::chrono::milliseconds(50), [] {
                return !frameQueue.empty() || stopFlag.load();
            });
            if (stopFlag && frameQueue.empty())
                break;
            if (frameQueue.empty())
                continue;
            frame = std::move(frameQueue.front());
            frameQueue.pop();
        }

        processAFrame(frame);
    }
}

void controlThreadFunc() {
    auto next_time = std::chrono::steady_clock::now();

    auto last_print_time = std::chrono::steady_clock::now();
    dart_vision::SendPacket packet;
    while (!stopFlag) {
        control_frame_count++;

        auto now = std::chrono::steady_clock::now();
        auto diff_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print_time).count();
        if (diff_ms >= 1000) {
            std::cout << "capture FPS: " << capture_frame_count
                      << " process FPS: " << process_frame_count
                      << " control FPS: " << control_frame_count << std::endl;
            control_frame_count = 0;
            process_frame_count = 0;
            capture_frame_count = 0;
            last_print_time = now;
        }
        cv::Point2f p;
        if (tracker && tracker->check()) {
            auto state = tracker->predict_future(now);
            p.x = state[0];
            p.y = state[2];
        }
        packet.distance_x = p.x-tracker->image_size_.width/2;
        packet.distance_y = p.y-tracker->image_size_.height/2;
        dart_vision::toZcbor(serial,packet);
        next_time += std::chrono::milliseconds(1);
        std::this_thread::sleep_until(next_time);
    }
}
void serialCallback(const uint8_t* data, std::size_t len) {}
void signalHandler(int signum) {
    std::cout << "\nSIGINT (Ctrl+C) detected. Stopping safely..." << std::endl;
    stopFlag = true;
}
int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);
    if (argc > 1) {
        std::string firstArg = argv[1];
        debug = (firstArg == "true" || firstArg == "1");
        std::cout << "debug: " << debug << std::endl;
    }
#ifdef ARM
    std::cout << "ARM" << std::endl;
#endif
#ifdef X86
    std::cout << "X86" << std::endl;
#endif
    std::string const path = "params.yaml";
    params.load(path);
#ifdef X86
    if (debug) {
        cv::namedWindow("frame", cv::WINDOW_NORMAL);
        cv::namedWindow("mask", cv::WINDOW_NORMAL);

        cv::createTrackbar("scale_percent %", "mask", &params.scale_percent, 100); // percent
        cv::createTrackbar("D_thresh %", "mask", &params.D_thresh, 50);
    }
#endif
    bool use_serial = params.use_serial;
    if (use_serial) {
        dart_vision::SerialDriver::SerialPortConfig cfg {
            /*baud*/ 115200,
            /*csize*/ 8,
            asio::serial_port_base::parity::none,
            asio::serial_port_base::stop_bits::one,
            asio::serial_port_base::flow_control::none
        };
        serial.init_port(params.serial_device, cfg);
        serial.set_receive_callback(std::bind(
            &serialCallback,
            std::placeholders::_1,
            std::placeholders::_2

        ));
        std::cout << "Serial port opened" << std::endl;
        serial.set_error_callback([&](const asio::error_code& ec) {
            std::cout << "serial error: " << ec.message() << std::endl;
        });
    }
    serial.start();
    tracker = std::make_unique<dart_vision::Tracker>(params);
    use_saver = params.use_saver;
    if (use_saver) {
        saver = std::make_unique<dart_vision::VideoSaver>(
            dart_vision::getTimestampFilename(),
            dart_vision::WriteMode::ASYNC
        );
        saver->start();
    }
    std::thread captureThread(captureThreadFunc);
    std::thread processThread(processThreadFunc);
    std::thread controlThread(controlThreadFunc);
    captureThread.join();
    processThread.join();
    controlThread.join();
    return 0;
}
