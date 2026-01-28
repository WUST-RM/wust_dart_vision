#pragma once
#include "pixel.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace flying_cv_master {

struct Size {
    int width = 0;
    int height = 0;
};

struct Rect {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;

    Size size() const {
        return { w, h };
    }
    bool empty() const {
        return w <= 0 || h <= 0;
    }
};

template<PixelFormat PF>
struct ImageView {
    using Traits = PixelFormatTraits<PF>;
    using ChannelT = typename Traits::channel_type;

    static constexpr int channels = Traits::channels;
    static constexpr size_t pixel_bytes = channels * sizeof(ChannelT);

    ChannelT* data = nullptr;
    int width = 0;
    int height = 0;
    size_t stride = 0; // bytes per row

    Size size() const {
        return { width, height };
    }

    bool empty() const {
        return !data || width <= 0 || height <= 0;
    }

    bool is_contiguous() const {
        return stride == width * pixel_bytes;
    }

    ChannelT* row_ptr(int y) const {
        return reinterpret_cast<ChannelT*>(reinterpret_cast<uint8_t*>(data) + y * stride);
    }

    ImageView roi(int x, int y, int w, int h) const {
        if (x < 0) {
            w += x;
            x = 0;
        }
        if (y < 0) {
            h += y;
            y = 0;
        }

        if (x + w > width)
            w = width - x;
        if (y + h > height)
            h = height - y;

        if (w <= 0 || h <= 0)
            return {};

        ChannelT* ptr = reinterpret_cast<ChannelT*>(
            reinterpret_cast<uint8_t*>(data) + y * stride + x * pixel_bytes
        );

        return { ptr, w, h, stride };
    }

    ImageView roi(Rect r) const {
        return roi(r.x, r.y, r.w, r.h);
    }
};

struct ImageControlBlock {
    uint8_t* data = nullptr;
    size_t size = 0;

    explicit ImageControlBlock(size_t sz): size(sz) {
        data = new uint8_t[size];
    }

    ~ImageControlBlock() {
        delete[] data;
    }
};

template<PixelFormat PF>
struct ImageBuffer {
    using Traits = PixelFormatTraits<PF>;
    using ChannelT = typename Traits::channel_type;

    static constexpr int channels = Traits::channels;
    static constexpr size_t pixel_bytes = channels * sizeof(ChannelT);

    int width = 0;
    int height = 0;
    size_t stride = 0;

    std::shared_ptr<ImageControlBlock> ctrl;

    ImageBuffer() = default;

    ImageBuffer(int w, int h): width(w), height(h) {
        stride = width * pixel_bytes;
        ctrl = std::make_shared<ImageControlBlock>(stride * height);
    }

    ImageView<PF> view() {
        return { reinterpret_cast<ChannelT*>(ctrl ? ctrl->data : nullptr), width, height, stride };
    }

    ImageView<PF> view() const {
        return { reinterpret_cast<ChannelT*>(ctrl ? ctrl->data : nullptr), width, height, stride };
    }

    ImageBuffer clone() const {
        ImageBuffer out(width, height);
        std::memcpy(out.ctrl->data, ctrl->data, stride * height);
        return out;
    }

    bool unique() const {
        return ctrl && ctrl.use_count() == 1;
    }

    void detach() {
        if (!ctrl || unique())
            return;
        *this = clone();
    }

    bool empty() const {
        return !ctrl || !ctrl->data;
    }
};

} // namespace flying_cv_master
