#pragma once
#include "image.hpp"
#include <cassert>

#ifdef __ARM_NEON
    #include <arm_neon.h>
#endif

namespace flying_cv_master {

inline void color_diff_mask(
    ImageView<PixelFormat::RGB> src,
    ImageView<PixelFormat::Grayscale> mask,
    uint8_t thresh
) {
    assert(!src.empty());
    assert(!mask.empty());
    assert(src.width == mask.width);
    assert(src.height == mask.height);

    const int w = src.width;
    const int h = src.height;

#ifdef __ARM_NEON
    const uint8x8_t vth = vdup_n_u8(thresh);
    const uint8x8_t v255 = vdup_n_u8(255);
#endif

    for (int y = 0; y < h; ++y) {
        const uint8_t* s = reinterpret_cast<const uint8_t*>(src.data) + y * src.stride;
        uint8_t* m = mask.data + y * mask.stride;

        int x = 0;

#ifdef __ARM_NEON
        for (; x <= w - 8; x += 8) {
            uint8x8x3_t rgb = vld3_u8(s + x * 3);

            uint8x8_t r = rgb.val[0];
            uint8x8_t g = rgb.val[1];
            uint8x8_t b = rgb.val[2];

            uint8x8_t max_rb = vmax_u8(r, b);

            // 饱和减法：diff = max(g - max_rb, 0)
            uint8x8_t diff = vqsub_u8(g, max_rb);

            // diff > thresh ?
            uint8x8_t cmp = vcgt_u8(diff, vth);

            // mask = 255 or 0
            uint8x8_t out = vand_u8(cmp, v255);

            vst1_u8(m + x, out);
        }
#endif
        for (; x < w; ++x) {
            uint8_t r = s[3 * x + 0];
            uint8_t g = s[3 * x + 1];
            uint8_t b = s[3 * x + 2];

            uint8_t max_rb = (r > b) ? r : b;
            uint8_t diff = (g > max_rb) ? (g - max_rb) : 0;

            m[x] = (diff > thresh) ? 255 : 0;
        }
    }
}

} // namespace flying_cv_master
