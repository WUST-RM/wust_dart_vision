#include "flying_cv_master/camera_v4l2.hpp"
#include "flying_cv_master/cv_algorithm.hpp"
#include "flying_cv_master/debug.hpp"
#include "utils.hpp"
#include <iostream>
using namespace flying_cv_master;
int main() {
    V4L2Camera cam("/dev/video0", 640, 480);

    ImageBuffer<PixelFormat::RGB> img;
    int count = 0;
    while (true) {
        img = cam.capture<PixelFormat::RGB>();
        if (img.empty())
            continue;
        auto view = img.view();
        ImageBuffer<PixelFormat::Grayscale> mask_buffer(view.width, view.height);
        auto mark = mask_buffer.view();
        color_diff_mask(view, mark, 30);
        // writeFrameToShm<PixelFormat::Grayscale>(mark);
        writeFrameToShm<PixelFormat::RGB>(view);
        count++;
        XSecOnce(
            [&]() {
                std::cout << "fps: " << count << std::endl;
                count = 0;
            },
            1.0
        );
    }
}