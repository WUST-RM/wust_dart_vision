#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace flying_cv_master {

enum class PixelFormat : int {
    Binary = 0,
    Grayscale = 1,
    RGB = 2,
};

template<PixelFormat>
struct PixelFormatTraits;

template<>
struct PixelFormatTraits<PixelFormat::Binary> {
    using channel_type = uint8_t; // physical storage
    static constexpr int channels = 1;
    static constexpr int bits_per_channel = 1;
    static constexpr bool is_interleaved = false;
    static constexpr bool bit_packed = true;
};

template<>
struct PixelFormatTraits<PixelFormat::Grayscale> {
    using channel_type = uint8_t;
    static constexpr int channels = 1;
    static constexpr int bits_per_channel = 8;
    static constexpr bool is_interleaved = false;
    static constexpr bool bit_packed = false;
};

template<>
struct PixelFormatTraits<PixelFormat::RGB> {
    using channel_type = uint8_t;
    static constexpr int channels = 3;
    static constexpr int bits_per_channel = 8;
    static constexpr bool is_interleaved = true;
    static constexpr bool bit_packed = false;
};


template<typename T, int C>
struct Pixel {
    T v[C];

    T& operator[](int i) {
        return v[i];
    }
    const T& operator[](int i) const {
        return v[i];
    }
};
struct RGBRef {
    uint8_t& r;
    uint8_t& g;
    uint8_t& b;
};
} // namespace flying_cv_master
