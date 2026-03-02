#pragma once
#include <chrono>
#include <fcntl.h>
#include <opencv2/opencv.hpp>
#ifdef ARM
    #include <arm_neon.h>
#endif
#include <sys/mman.h>
#include <sys/stat.h>
namespace dart_vision {
template<typename Func>
void XSecOnce(Func&& func, double dt) noexcept {
    static auto last_call = std::chrono::steady_clock::now();

    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - last_call).count();

    if (elapsed >= dt) {
        last_call = now;
        func();
    }
}
inline void writeFrameToShm(const cv::Mat& frame) {
    static const char* shm_name = "/debug_frame";
    static const size_t shm_max_size = 2 * 1024 * 1024; // 2MB
    static auto last_show_time = std::chrono::steady_clock::now();

    if (frame.empty())
        return;

    auto now = std::chrono::steady_clock::now();
    const double min_interval_ms = 1000.0 / 10.0; // 10 FPS limit
    if (std::chrono::duration<double, std::milli>(now - last_show_time).count() < min_interval_ms)
        return;
    last_show_time = now;
    std::vector<uchar> buf;
    std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 70 }; // JPEG 质量
    cv::Mat frame_rgb;

    frame_rgb = frame;

    if (!cv::imencode(".jpg", frame_rgb, buf, params)) {
        std::cerr << "[writeFrameToShm] JPEG encoding failed\n";
        return;
    }

    if (buf.size() > shm_max_size - 4) {
        std::cerr << "[writeFrameToShm] JPEG too large: " << buf.size() << "\n";
        return;
    }
    int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        perror("shm_open");
        return;
    }
    if (ftruncate(fd, shm_max_size) == -1) {
        perror("ftruncate");
        close(fd);
        return;
    }
    void* ptr = mmap(nullptr, shm_max_size, PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return;
    }

    uint32_t size_u32 = static_cast<uint32_t>(buf.size());
    memcpy(ptr, &size_u32, sizeof(uint32_t));
    memcpy((char*)ptr + 4, buf.data(), buf.size());
    munmap(ptr, shm_max_size);
    close(fd);
}
#ifdef ARM
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
#endif

} // namespace dart_vision