#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
namespace dart_vision {
// -------------------- 生成时间戳文件名 --------------------
inline std::string getTimestampFilename(const std::string& ext = ".avi") {
    using namespace std::chrono;

    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);

    std::tm tm {};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str() + ext;
}

// -------------------- 写入模式 --------------------
enum class WriteMode {
    ASYNC, // 异步写入：后台线程写入，队列满则丢帧
    BLOCKING // 阻塞写入：直接写磁盘，不卡队列，不丢帧
};

// -------------------- VideoSaver --------------------
class VideoSaver {
public:
    VideoSaver(
        const std::string& filename,
        WriteMode mode = WriteMode::ASYNC,
        int fps = 60,
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        size_t max_queue_size = 20
    ):
        filename_(filename),
        mode_(mode),
        fps_(fps),
        fourcc_(fourcc),
        max_queue_size_(max_queue_size),
        running_(false),
        videoOpened_(false) {}

    ~VideoSaver() {
        stop(); // 安全析构自动保存
    }

    // -------------------- 启动视频保存 --------------------
    void start() {
        if (running_)
            return;
        running_ = true;

        if (mode_ == WriteMode::ASYNC) {
            worker_ = std::thread(&VideoSaver::run, this);
        }
    }

    // -------------------- 停止保存并 flush --------------------
    void stop() {
        if (!running_)
            return;

        running_ = false;
        cond_.notify_all();

        if (mode_ == WriteMode::ASYNC && worker_.joinable())
            worker_.join();

        if (videoOpened_) {
            writer_.release();
            videoOpened_ = false;
        }

        std::cout << "Video saved: " << filename_ << std::endl;
    }

    // -------------------- 推送帧 --------------------
    void pushFrame(const cv::Mat& frame) {
        if (!running_)
            return;

        // -------- 阻塞写入模式 --------
        if (mode_ == WriteMode::BLOCKING) {
            if (!videoOpened_) {
                writer_.open(filename_, fourcc_, fps_, frame.size());
                if (!writer_.isOpened()) {
                    std::cerr << "Error opening video writer (blocking): " << filename_
                              << std::endl;
                    return;
                }
                videoOpened_ = true;
            }
            writer_.write(frame); // 同步写入
            return;
        }

        // -------- 异步模式 --------
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (queue_.size() >= max_queue_size_) {
                queue_.pop(); // 丢旧帧
            }
            queue_.push(std::move(frame));
        }
        cond_.notify_one();
    }

private:
    // -------------------- 异步写入线程函数 --------------------
    void run() {
        while (true) {
            cv::Mat frame;

            {
                std::unique_lock<std::mutex> lk(mtx_);
                cond_.wait(lk, [&] { return !queue_.empty() || !running_; });

                if (!running_ && queue_.empty())
                    break;

                frame = queue_.front();
                queue_.pop();
            }

            if (!videoOpened_) {
                writer_.open(filename_, fourcc_, fps_, frame.size());
                if (!writer_.isOpened()) {
                    std::cerr << "Error opening video writer (async): " << filename_ << std::endl;
                    return;
                }
                videoOpened_ = true;
            }

            writer_.write(frame);
        }
    }

private:
    std::string filename_;
    WriteMode mode_;
    int fps_;
    int fourcc_;
    size_t max_queue_size_;

    std::queue<cv::Mat> queue_;
    std::mutex mtx_;
    std::condition_variable cond_;

    cv::VideoWriter writer_;
    std::thread worker_;
    std::atomic<bool> running_;
    bool videoOpened_;
};
} // namespace dart_vision