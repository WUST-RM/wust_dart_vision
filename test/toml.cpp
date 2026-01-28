#include <iostream>
#include <toml++/toml.h>

int main() {
    toml::table tbl = toml::parse_file("/home/hy/RV1106/flying_cv_master/config.toml");

    int width = tbl["camera"]["width"].value_or(640);
    int height = tbl["camera"]["height"].value_or(480);
    int fps = tbl["camera"]["fps"].value_or(30);

    std::string dev = tbl["camera"]["device"].value_or("/dev/video0");

    bool enable = tbl["detector"]["enable"].value_or(false);

    std::cout << width << " " << height << " " << fps << "\n";
}
