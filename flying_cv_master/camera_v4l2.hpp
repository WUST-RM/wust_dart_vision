#pragma once

#include <algorithm>
#include <fcntl.h>
#include <iostream>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "image.hpp"

namespace flying_cv_master {
template<PixelFormat PF>
struct V4L2PixelTraits;

template<>
struct V4L2PixelTraits<PixelFormat::Grayscale> {
    static constexpr uint32_t v4l2_fmt = V4L2_PIX_FMT_GREY;
};

template<>
struct V4L2PixelTraits<PixelFormat::RGB> {
    static constexpr uint32_t v4l2_fmt = V4L2_PIX_FMT_YUYV;
    static constexpr bool need_convert = true;
};
template<>
struct V4L2PixelTraits<PixelFormat::Binary> {
    static constexpr uint32_t v4l2_fmt = V4L2_PIX_FMT_GREY;
};
inline void yuyv_to_gray(const uint8_t* src, uint8_t* dst, int w, int h, size_t src_stride) {
    for (int y = 0; y < h; ++y) {
        const uint8_t* s = src + y * src_stride;
        uint8_t* d = dst + y * w;
        for (int x = 0; x < w; ++x) {
            d[x] = s[x * 2]; // Y
        }
    }
}

inline void yuyv_to_rgb(const uint8_t* src, uint8_t* dst, int w, int h, size_t src_stride) {
    for (int y = 0; y < h; ++y) {
        const uint8_t* s = src + y * src_stride;
        uint8_t* d = dst + y * w * 3;

        for (int x = 0; x < w; x += 2) {
            int y0 = s[0], u = s[1] - 128;
            int y1 = s[2], v = s[3] - 128;

            auto conv = [&](int Y, uint8_t* p) {
                int r = Y + 1.402 * v;
                int g = Y - 0.344 * u - 0.714 * v;
                int b = Y + 1.772 * u;
                p[0] = std::clamp(r, 0, 255);
                p[1] = std::clamp(g, 0, 255);
                p[2] = std::clamp(b, 0, 255);
            };

            conv(y0, d);
            conv(y1, d + 3);

            s += 4;
            d += 6;
        }
    }
}

class V4L2Camera {
public:
    V4L2Camera(const std::string& device, int w, int h): width_(w), height_(h) {
        if (!open_device(device)) {
            bool found = false;
            for (int i = 0; i < 100; i++) {
                std::string try_device = "/dev/video" + std::to_string(i);
                if (open_device(try_device)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw std::runtime_error("Cannot open camera device");
            }
        }
    }

    ~V4L2Camera() {
        stop_stream();
        close_device();
    }

    template<PixelFormat PF>
    ImageBuffer<PF> capture() {
        ensure_format<PF>();

        v4l2_buffer buf {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        ioctl_or_throw(VIDIOC_DQBUF, &buf);

        ImageBuffer<PF> out(width_, height_);
        auto view = out.view();

        if constexpr (!V4L2PixelTraits<PF>::need_convert) {
            std::memcpy(view.data, buffers_[buf.index].start, view.stride * view.height);
        } else {
            if constexpr (PF == PixelFormat::Grayscale) {
                yuyv_to_gray(
                    buffers_[buf.index].start,
                    view.data,
                    width_,
                    height_,
                    buffers_[buf.index].length / height_
                );
            } else if constexpr (PF == PixelFormat::RGB) {
                yuyv_to_rgb(
                    buffers_[buf.index].start,
                    view.data,
                    width_,
                    height_,
                    buffers_[buf.index].length / height_
                );
            }
        }

        ioctl_or_throw(VIDIOC_QBUF, &buf);
        return out;
    }

private:
    int fd_ { -1 };
    int width_ { 0 }, height_ { 0 };
    uint32_t current_fmt_ { 0 };

    struct Buffer {
        uint8_t* start;
        size_t length;
    };
    std::vector<Buffer> buffers_;

    template<PixelFormat PF>
    void ensure_format() {
        constexpr uint32_t want = V4L2PixelTraits<PF>::v4l2_fmt;
        if (current_fmt_ == want)
            return;

        stop_stream();
        init_device(want);
        start_stream();
        current_fmt_ = want;
    }

    void init_device(uint32_t fmt) {
        v4l2_format f {};
        f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        f.fmt.pix.width = width_;
        f.fmt.pix.height = height_;
        f.fmt.pix.pixelformat = fmt;
        ioctl_or_throw(VIDIOC_S_FMT, &f);

        v4l2_requestbuffers req {};
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        ioctl_or_throw(VIDIOC_REQBUFS, &req);

        buffers_.resize(req.count);
        for (size_t i = 0; i < req.count; ++i) {
            v4l2_buffer b {};
            b.type = req.type;
            b.memory = req.memory;
            b.index = i;
            ioctl_or_throw(VIDIOC_QUERYBUF, &b);

            buffers_[i].length = b.length;
            buffers_[i].start = static_cast<uint8_t*>(
                mmap(nullptr, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, b.m.offset)
            );

            ioctl_or_throw(VIDIOC_QBUF, &b);
        }
    }

    bool open_device(const std::string& dev) {
        fd_ = open(dev.c_str(), O_RDWR);
        if (fd_ < 0) {
            std::cout << "open_device failed: " << dev << std::endl;
            return false;
        }
        return true;
    }

    void start_stream() {
        v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl_or_throw(VIDIOC_STREAMON, &t);
    }

    void stop_stream() {
        if (fd_ < 0)
            return;
        v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(fd_, VIDIOC_STREAMOFF, &t);
    }

    void close_device() {
        for (auto& b: buffers_)
            munmap(b.start, b.length);
        if (fd_ >= 0)
            close(fd_);
    }

    void ioctl_or_throw(unsigned long r, void* a) {
        if (ioctl(fd_, r, a) < 0)
            throw std::runtime_error("V4L2 ioctl failed");
    }
};

} // namespace flying_cv_master
