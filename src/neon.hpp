// neon_diff_threshold_opt.hpp
#pragma once
#include <arm_neon.h>
#include <cstdint>
#include <opencv2/opencv.hpp>
namespace dart_vision {
// ----------------------------------------------
//  NEON optimized BGR -> CV_8UC1 mask
// ----------------------------------------------
inline void neon_diff_threshold_bgr(const cv::Mat& src, cv::Mat& dst, uint8_t D_thresh) {
    CV_Assert(src.type() == CV_8UC3);
    CV_Assert(dst.type() == CV_8UC1);
    const int w = src.cols;
    const int h = src.rows;

    const uint8x16_t threshv = vdupq_n_u8(D_thresh);

    // Continuous memory fast-path
    if (src.isContinuous() && dst.isContinuous()) {
        const uint8_t* ps = src.data;
        uint8_t* pd = dst.data;
        size_t total = static_cast<size_t>(w) * h;
        size_t i = 0;

        // Process 32 pixels per iteration (2 x 16)
        for (; i + 31 < total; i += 32) {
            uint8x16x3_t v1 = vld3q_u8(ps);
            uint8x16x3_t v2 = vld3q_u8(ps + 48);

            uint8x16_t max1 = vmaxq_u8(v1.val[2], v1.val[0]);
            uint8x16_t max2 = vmaxq_u8(v2.val[2], v2.val[0]);

            uint8x16_t diff1 = vqsubq_u8(v1.val[1], max1);
            uint8x16_t diff2 = vqsubq_u8(v2.val[1], max2);

            vst1q_u8(pd, vcgtq_u8(diff1, threshv));
            vst1q_u8(pd + 16, vcgtq_u8(diff2, threshv));

            ps += 96; // 32 pixels * 3 bytes
            pd += 32;
        }

        // leftover 16 pixels
        for (; i + 15 < total; i += 16) {
            uint8x16x3_t v = vld3q_u8(ps);
            uint8x16_t maxv = vmaxq_u8(v.val[2], v.val[0]);
            uint8x16_t diff = vqsubq_u8(v.val[1], maxv);
            vst1q_u8(pd, vcgtq_u8(diff, threshv));
            ps += 48;
            pd += 16;
        }

        // tail
        for (; i < total; ++i) {
            uint8_t B = ps[0], G = ps[1], R = ps[2];
            uint8_t maxRB = (R > B) ? R : B;
            uint8_t diff = (G > maxRB) ? (G - maxRB) : 0;
            *pd = (diff > D_thresh) ? 255 : 0;
            ps += 3;
            pd += 1;
        }
        return;
    }

    // Row-by-row fallback
    for (int y = 0; y < h; ++y) {
        const uint8_t* ps = src.data + y * src.step;
        uint8_t* pd = dst.data + y * dst.step;
        int x = 0;

        for (; x + 31 < w; x += 32) {
            uint8x16x3_t v1 = vld3q_u8(ps);
            uint8x16x3_t v2 = vld3q_u8(ps + 48);

            uint8x16_t max1 = vmaxq_u8(v1.val[2], v1.val[0]);
            uint8x16_t max2 = vmaxq_u8(v2.val[2], v2.val[0]);

            uint8x16_t diff1 = vqsubq_u8(v1.val[1], max1);
            uint8x16_t diff2 = vqsubq_u8(v2.val[1], max2);

            vst1q_u8(pd, vcgtq_u8(diff1, threshv));
            vst1q_u8(pd + 16, vcgtq_u8(diff2, threshv));

            ps += 96;
            pd += 32;
        }

        for (; x + 15 < w; x += 16) {
            uint8x16x3_t v = vld3q_u8(ps);
            uint8x16_t maxv = vmaxq_u8(v.val[2], v.val[0]);
            uint8x16_t diff = vqsubq_u8(v.val[1], maxv);
            vst1q_u8(pd, vcgtq_u8(diff, threshv));
            ps += 48;
            pd += 16;
        }

        for (; x < w; ++x) {
            uint8_t B = ps[0], G = ps[1], R = ps[2];
            uint8_t maxRB = (R > B) ? R : B;
            uint8_t diff = (G > maxRB) ? (G - maxRB) : 0;
            *pd = (diff > D_thresh) ? 255 : 0;
            ps += 3;
            pd++;
        }
    }
}

// ----------------------------------------------
// BGRA -> mask optimized
// ----------------------------------------------
inline void neon_diff_threshold_bgra(const cv::Mat& src, cv::Mat& dst, uint8_t D_thresh) {
    CV_Assert(src.type() == CV_8UC4);
    CV_Assert(dst.type() == CV_8UC1);
    const int w = src.cols;
    const int h = src.rows;

    const uint8x16_t threshv = vdupq_n_u8(D_thresh);

    for (int y = 0; y < h; ++y) {
        const uint8_t* ps = src.data + y * src.step;
        uint8_t* pd = dst.data + y * dst.step;
        int x = 0;

        for (; x + 15 < w; x += 16) {
            uint8x16x4_t v = vld4q_u8(ps);
            uint8x16_t maxv = vmaxq_u8(v.val[2], v.val[0]);
            uint8x16_t diff = vqsubq_u8(v.val[1], maxv);
            vst1q_u8(pd, vcgtq_u8(diff, threshv));
            ps += 64;
            pd += 16; // 16 * 4
        }

        for (; x < w; ++x) {
            uint8_t B = ps[0], G = ps[1], R = ps[2];
            uint8_t maxRB = (R > B) ? R : B;
            uint8_t diff = (G > maxRB) ? (G - maxRB) : 0;
            *pd = (diff > D_thresh) ? 255 : 0;
            ps += 4;
            pd++;
        }
    }
}

// ----------------------------------------------
// Dispatcher
// ----------------------------------------------
inline void neon_diff_threshold_dispatch(const cv::Mat& src, cv::Mat& dst, uint8_t D_thresh) {
    CV_Assert(dst.rows == src.rows && dst.cols == src.cols && dst.type() == CV_8UC1);

    if (src.type() == CV_8UC3) {
        neon_diff_threshold_bgr(src, dst, D_thresh);
    } else if (src.type() == CV_8UC4) {
        neon_diff_threshold_bgra(src, dst, D_thresh);
    } else {
        cv::Mat tmp;
        cv::cvtColor(src, tmp, cv::COLOR_BGRA2BGR);
        neon_diff_threshold_bgr(tmp, dst, D_thresh);
    }
}
} // namespace dart_vision