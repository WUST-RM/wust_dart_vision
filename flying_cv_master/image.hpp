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

    bool empty() const {
        return w <= 0 || h <= 0;
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
struct ImageView {
    using Traits = PixelFormatTraits<PF>;
    static_assert(!Traits::bit_packed, "Binary has its own specialization");

    using ChannelT = typename Traits::channel_type;

    static constexpr int channels = Traits::channels;
    static constexpr size_t pixel_bytes = channels * sizeof(ChannelT);

    ChannelT* data = nullptr;
    int width = 0;
    int height = 0;
    size_t stride = 0; // bytes per row

    bool empty() const {
        return !data || width <= 0 || height <= 0;
    }

    bool is_contiguous() const {
        return stride == width * pixel_bytes;
    }

    ChannelT* row_ptr(int y) const {
        return reinterpret_cast<ChannelT*>(
            reinterpret_cast<uint8_t*>(data) + y * stride
        );
    }

    ImageView roi(int x, int y, int w, int h) const {
        if (x < 0) { w += x; x = 0; }
        if (y < 0) { h += y; y = 0; }
        if (x + w > width)  w = width - x;
        if (y + h > height) h = height - y;
        if (w <= 0 || h <= 0) return {};

        ChannelT* ptr = reinterpret_cast<ChannelT*>(
            reinterpret_cast<uint8_t*>(data) + y * stride + x * pixel_bytes
        );
        return { ptr, w, h, stride };
    }

    ImageView roi(Rect r) const {
        return roi(r.x, r.y, r.w, r.h);
    }
};



template<>
struct ImageView<PixelFormat::Binary> {
    using Traits = PixelFormatTraits<PixelFormat::Binary>;

    uint8_t* data = nullptr;   // bit-packed
    int width = 0;
    int height = 0;
    size_t stride = 0;         // bytes per row

    bool empty() const {
        return !data || width <= 0 || height <= 0;
    }

    uint8_t* row_ptr(int y) const {
        return data + y * stride;
    }
};


inline bool binary_get(
    const ImageView<PixelFormat::Binary>& img,
    int x, int y
) {
    assert(x >= 0 && x < img.width);
    assert(y >= 0 && y < img.height);

    const uint8_t* row = img.row_ptr(y);
    return (row[x >> 3] >> (7 - (x & 7))) & 1;
}

inline void binary_set(
    ImageView<PixelFormat::Binary>& img,
    int x, int y,
    bool v
) {
    assert(x >= 0 && x < img.width);
    assert(y >= 0 && y < img.height);

    uint8_t* row = img.row_ptr(y);
    uint8_t& byte = row[x >> 3];
    uint8_t mask = 1u << (7 - (x & 7));

    if (v)
        byte |= mask;
    else
        byte &= ~mask;
}

template<PixelFormat PF>
struct ImageBuffer {
    using Traits = PixelFormatTraits<PF>;
    static_assert(!Traits::bit_packed, "Binary has its own specialization");

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
        return {
            reinterpret_cast<ChannelT*>(ctrl ? ctrl->data : nullptr),
            width,
            height,
            stride
        };
    }

    ImageView<PF> view() const {
        return {
            reinterpret_cast<ChannelT*>(ctrl ? ctrl->data : nullptr),
            width,
            height,
            stride
        };
    }

    ImageBuffer clone() const {
        ImageBuffer out(width, height);
        std::memcpy(out.ctrl->data, ctrl->data, stride * height);
        return out;
    }

    bool empty() const {
        return !ctrl || !ctrl->data;
    }
};



template<>
struct ImageBuffer<PixelFormat::Binary> {
    int width = 0;
    int height = 0;
    size_t stride = 0; // bytes per row

    std::shared_ptr<ImageControlBlock> ctrl;

    ImageBuffer() = default;

    ImageBuffer(int w, int h): width(w), height(h) {
        stride = (w + 7) / 8;
        ctrl = std::make_shared<ImageControlBlock>(stride * height);
        std::memset(ctrl->data, 0, stride * height);
    }

    ImageView<PixelFormat::Binary> view() {
        return { ctrl ? ctrl->data : nullptr, width, height, stride };
    }

    ImageView<PixelFormat::Binary> view() const {
        return { ctrl ? ctrl->data : nullptr, width, height, stride };
    }

    bool empty() const {
        return !ctrl || !ctrl->data;
    }
};

} // namespace flying_cv_master
