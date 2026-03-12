#include <uvc.hpp>
using namespace dart_vision;

int main() {
    auto config = toml::parse_file("/home/hy/RV1106/dart_vision/config.toml");
    auto uvc_table = config["uvc"].as_table();
    UVC uvc(*uvc_table);

    return 0;
}