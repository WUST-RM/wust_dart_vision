#pragma once
#include "image.hpp"

#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace flying_cv_master {

struct ShmImageHeader {
    uint32_t magic = 0x494D4756; // 'IMGV'
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t pixel_fmt;
    uint32_t data_size;
};

template<PixelFormat PF>
void writeFrameToShm(const ImageView<PF>& img) {
    static constexpr const char* shm_name = "/debug_frame";
    static constexpr size_t shm_max_size = 8 * 1024 * 1024; // 8MB

    if (img.empty())
        return;

    const size_t pixel_bytes = img.stride * img.height;
    const size_t total_size = sizeof(ShmImageHeader) + pixel_bytes;

    if (total_size > shm_max_size) {
        std::cout << "[writeFrameToShm] frame too large\n";
        return;
    }

    int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (fd < 0)
        return;

    const auto unuse = ftruncate(fd, shm_max_size);

    void* ptr = mmap(nullptr, shm_max_size, PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        std::cout << "[writeFrameToShm] mmap failed\n";
        return;
    }

    auto* hdr = reinterpret_cast<ShmImageHeader*>(ptr);
    hdr->magic = 0x494D4756;
    hdr->width = img.width;
    hdr->height = img.height;
    hdr->stride = img.stride;
    hdr->pixel_fmt = static_cast<uint32_t>(PF);
    hdr->data_size = pixel_bytes;

    std::memcpy(static_cast<uint8_t*>(ptr) + sizeof(ShmImageHeader), img.data, pixel_bytes);

    munmap(ptr, shm_max_size);
    close(fd);
}

} // namespace flying_cv_master
