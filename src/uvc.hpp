#pragma once

#include <opencv2/opencv.hpp>
#include <toml++/toml.h>
#include <turbojpeg.h> // <-- libjpeg-turbo

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace dart_vision {

class UVC {
public:
    using Ptr = std::unique_ptr<UVC>;

    struct Buffer {
        void* start = nullptr;
        size_t length = 0;
    };

    UVC(const toml::table& config) {
        device_name_ = config["device_name"].value_or("");
        fps_ = config["fps"].value_or(30);
        width_ = config["width"].value_or(640);
        height_ = config["height"].value_or(480);

        // 新增曝光、增益、伽马参数
        exposure_ = config["exposure"].value_or(10.0);
        gain_ = config["gain"].value_or(10.0);
        gamma_ = config["gamma"].value_or(100.0);

        std::cout << "UVC loaded: " << device_name_ << " " << width_ << "x" << height_ << "@"
                  << fps_ << ", exposure=" << exposure_ << ", gain=" << gain_
                  << ", gamma=" << gamma_ << std::endl;

        openDevice();
        initDevice();
        startStream();
        setCameraParams(); // 设置曝光、增益、伽马

        // initialize turbojpeg
        tj_handle_ = tjInitDecompress();
        if (!tj_handle_) {
            throw std::runtime_error("tjInitDecompress failed");
        }
    }

    ~UVC() {
        if (tj_handle_) {
            tjDestroy(tj_handle_);
        }

        if (fd_ >= 0) {
            int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ioctl(fd_, VIDIOC_STREAMOFF, &type);
            for (auto& b: buffers_) {
                if (b.start)
                    munmap(b.start, b.length);
            }
            close(fd_);
        }
    }

    static Ptr create(const toml::table& config) {
        return std::make_unique<UVC>(config);
    }

    cv::Mat read(int timeout_ms = 200) {
        struct pollfd pfd;
        pfd.fd = fd_;
        pfd.events = POLLIN | POLLPRI;
        int ret = poll(&pfd, 1, timeout_ms);
        if (ret < 0) {
            if (errno == EINTR)
                return {};
            std::cerr << "poll error: " << strerror(errno) << std::endl;
            return {};
        }
        if (ret == 0)
            return {}; // timeout

        v4l2_buffer buf {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_, VIDIOC_DQBUF, &buf) == -1) {
            if (errno == EAGAIN)
                return {};
            std::cerr << "VIDIOC_DQBUF error: " << strerror(errno) << std::endl;
            return {};
        }

        cv::Mat ret_mat;
        if (pixelformat_ == V4L2_PIX_FMT_MJPEG) {
            unsigned long jpeg_size = buf.bytesused;
            unsigned char* jpeg_buf = reinterpret_cast<unsigned char*>(buffers_[buf.index].start);

            int tj_ret = tjDecompress2(
                tj_handle_,
                jpeg_buf,
                jpeg_size,
                out_mat_.data,
                width_,
                0,
                height_,
                TJPF_BGR,
                TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE
            );
            if (tj_ret != 0) {
                std::cerr << "tjDecompress2 failed: " << tjGetErrorStr() << std::endl;
                ioctl(fd_, VIDIOC_QBUF, &buf);
                return {};
            }
            ret_mat = out_mat_.clone();
        } else if (pixelformat_ == V4L2_PIX_FMT_YUYV) {
            cv::Mat yuyv(height_, width_, CV_8UC2, buffers_[buf.index].start);
            if (out_mat_.empty() || out_mat_.cols != width_ || out_mat_.rows != height_)
                out_mat_.create(height_, width_, CV_8UC3);
            cv::cvtColor(yuyv, out_mat_, cv::COLOR_YUV2BGR_YUYV);
            ret_mat = out_mat_.clone();
        } else if (pixelformat_ == V4L2_PIX_FMT_GREY) {
            cv::Mat grey(height_, width_, CV_8UC1, buffers_[buf.index].start);
            if (out_mat_.empty() || out_mat_.cols != width_ || out_mat_.rows != height_)
                out_mat_.create(height_, width_, CV_8UC3);
            cv::cvtColor(grey, out_mat_, cv::COLOR_GRAY2BGR);
            ret_mat = out_mat_.clone();
        }

        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            std::cerr << "VIDIOC_QBUF failed: " << strerror(errno) << std::endl;
        }

        last_frame_time_ = std::chrono::steady_clock::now();
        return ret_mat;
    }

public:
    std::string device_name_;
    int fps_;
    int width_;
    int height_;
    double exposure_; // 曝光
    double gain_; // 增益
    double gamma_; // 伽马
    std::chrono::steady_clock::time_point last_frame_time_;

private:
    int fd_ = -1;
    std::vector<Buffer> buffers_;
    __u32 pixelformat_ = V4L2_PIX_FMT_MJPEG;
    int req_buffer_count_ = 4;

    tjhandle tj_handle_ = nullptr;
    cv::Mat out_mat_; // persistent output Mat

private:
    void openDevice() {
        fd_ = open(device_name_.c_str(), O_RDWR | O_NONBLOCK, 0);
        if (fd_ < 0) {
            for (int i = 0; i < 10; ++i) {
                std::string dev = "/dev/video" + std::to_string(i);
                fd_ = open(dev.c_str(), O_RDWR | O_NONBLOCK, 0);
                if (fd_ >= 0) {
                    device_name_ = dev;
                    break;
                }
            }
        }
        if (fd_ < 0)
            throw std::runtime_error("Cannot open UVC device: " + device_name_);
    }

    void initDevice() {
        v4l2_format fmt {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width_;
        fmt.fmt.pix.height = height_;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;

        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0)
            throw std::runtime_error("VIDIOC_S_FMT failed: " + std::string(strerror(errno)));

        v4l2_format got {};
        got.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_G_FMT, &got) < 0)
            throw std::runtime_error("VIDIOC_G_FMT failed");
        pixelformat_ = got.fmt.pix.pixelformat;
        width_ = got.fmt.pix.width;
        height_ = got.fmt.pix.height;

        if (pixelformat_ == V4L2_PIX_FMT_MJPEG)
            out_mat_.create(height_, width_, CV_8UC3);

        v4l2_streamparm parm {};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator = 1;
        parm.parm.capture.timeperframe.denominator = fps_;
        ioctl(fd_, VIDIOC_S_PARM, &parm);

        v4l2_requestbuffers req {};
        req.count = req_buffer_count_;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0)
            throw std::runtime_error("VIDIOC_REQBUFS failed");

        buffers_.resize(req.count);
        for (size_t i = 0; i < buffers_.size(); ++i) {
            v4l2_buffer buf {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = static_cast<unsigned int>(i);
            if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0)
                throw std::runtime_error("VIDIOC_QUERYBUF failed");
            buffers_[i].length = buf.length;
            buffers_[i].start =
                mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
            if (buffers_[i].start == MAP_FAILED)
                throw std::runtime_error("mmap failed");
            if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0)
                throw std::runtime_error("VIDIOC_QBUF failed");
        }
    }

    void startStream() {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0)
            throw std::runtime_error("VIDIOC_STREAMON failed");
    }

    void setCameraParams() {
        // 曝光
        v4l2_control ctrl {};
        ctrl.id = V4L2_CID_EXPOSURE_ABSOLUTE;
        ctrl.value = static_cast<int>(exposure_);
        if (ioctl(fd_, VIDIOC_S_CTRL, &ctrl) < 0) {
            std::cerr << "Set exposure failed: " << strerror(errno) << std::endl;
        }

        // 增益
        ctrl.id = V4L2_CID_GAIN;
        ctrl.value = static_cast<int>(gain_);
        if (ioctl(fd_, VIDIOC_S_CTRL, &ctrl) < 0) {
            std::cerr << "Set gain failed: " << strerror(errno) << std::endl;
        }

        // 伽马
        ctrl.id = V4L2_CID_GAMMA;
        ctrl.value = static_cast<int>(gamma_);
        if (ioctl(fd_, VIDIOC_S_CTRL, &ctrl) < 0) {
            std::cerr << "Set gamma failed: " << strerror(errno) << std::endl;
        }
    }
};

} // namespace dart_vision