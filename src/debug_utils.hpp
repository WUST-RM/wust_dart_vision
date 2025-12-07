#pragma once
#include <chrono>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

// POSIX SHM
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <jpeglib.h>
#include <setjmp.h>

namespace dart_vision {
void writeFrameToShm(const cv::Mat& frame) {
    static const char* shm_name = "/debug_frame";
    static const size_t shm_max_size = 2 * 1024 * 1024; // 2MB
    static auto last_show_time = std::chrono::steady_clock::now();
    if (frame.empty())
        return;
    auto now = std::chrono::steady_clock::now();
    const double min_interval_ms = 1000.0 / 10.0;
    if (std::chrono::duration<double, std::milli>(now - last_show_time).count() < min_interval_ms)
        return;
    last_show_time = now;
    // JPEG output buffer
    unsigned char* jpegBuf = nullptr;
    unsigned long jpegSize = 0;

    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);

    jpeg_create_compress(&cinfo);
    jpeg_mem_dest(&cinfo, &jpegBuf, &jpegSize); // 输出到内存

    cinfo.image_width = frame.cols;
    cinfo.image_height = frame.rows;
    cinfo.input_components = frame.channels();
    if (frame.channels() == 1) {
        cinfo.in_color_space = JCS_GRAYSCALE;
    } else {
        cinfo.in_color_space = JCS_RGB; // OpenCV 默认 BGR, convert to RGB for libjpeg
    }

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 70, TRUE); // JPEG 质量（可调）

    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = (JSAMPLE*)&frame.data[cinfo.next_scanline * frame.step];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    // ---- Shared Memory ----
    if (jpegSize > shm_max_size - 4) {
        std::cerr << "[writeFrameToShm] JPEG too large: " << jpegSize << "\n";
        free(jpegBuf);
        return;
    }

    int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        perror("shm_open");
        free(jpegBuf);
        return;
    }
    ftruncate(fd, shm_max_size);

    void* ptr = mmap(nullptr, shm_max_size, PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        free(jpegBuf);
        return;
    }

    uint32_t size_u32 = static_cast<uint32_t>(jpegSize);
    memcpy(ptr, &size_u32, 4);
    memcpy((char*)ptr + 4, jpegBuf, jpegSize);

    munmap(ptr, shm_max_size);
    close(fd);
    free(jpegBuf);
}

} // namespace dart_vision
