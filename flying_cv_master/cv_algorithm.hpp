#pragma once
#include "image.hpp"
#include <cassert>

#ifdef __ARM_NEON
    #include <arm_neon.h>
#endif

namespace flying_cv_master {

inline void color_diff_mask(
    const ImageView<PixelFormat::RGB>& src,
    ImageView<PixelFormat::Binary>& dst,
    uint8_t thresh
) {
    assert(!src.empty());
    assert(!dst.empty());
    assert(src.width == dst.width);
    assert(src.height == dst.height);

    const int w = src.width;
    const int h = src.height;

#ifdef __ARM_NEON
    const uint8x8_t vth = vdup_n_u8(thresh);
#endif

    for (int y = 0; y < h; ++y) {
        const uint8_t* s =
            reinterpret_cast<const uint8_t*>(src.data) + y * src.stride;
        uint8_t* d = dst.row_ptr(y);

        int x = 0;

#ifdef __ARM_NEON
        for (; x <= w - 8; x += 8) {
            uint8x8x3_t rgb = vld3_u8(s + x * 3);

            uint8x8_t r = rgb.val[0];
            uint8x8_t g = rgb.val[1];
            uint8x8_t b = rgb.val[2];

            uint8x8_t max_rb = vmax_u8(r, b);
            uint8x8_t diff = vqsub_u8(g, max_rb);


            uint8x8_t cmp = vcgt_u8(diff, vth);


            uint8_t mask =
                ((vget_lane_u8(cmp, 0) >> 7) << 7) |
                ((vget_lane_u8(cmp, 1) >> 7) << 6) |
                ((vget_lane_u8(cmp, 2) >> 7) << 5) |
                ((vget_lane_u8(cmp, 3) >> 7) << 4) |
                ((vget_lane_u8(cmp, 4) >> 7) << 3) |
                ((vget_lane_u8(cmp, 5) >> 7) << 2) |
                ((vget_lane_u8(cmp, 6) >> 7) << 1) |
                ((vget_lane_u8(cmp, 7) >> 7) << 0);

            d[x >> 3] = mask;
        }
#endif

        for (; x < w; ++x) {
            uint8_t r = s[3 * x + 0];
            uint8_t g = s[3 * x + 1];
            uint8_t b = s[3 * x + 2];

            uint8_t max_rb = (r > b) ? r : b;
            uint8_t diff = (g > max_rb) ? (g - max_rb) : 0;

            binary_set(dst, x, y, diff > thresh);
        }
    }
}

} // namespace flying_cv_master
