#include "serial_driver.hpp"
#include <iostream>

int main() {
    dart_vision::SerialDriver serial;
    dart_vision::SerialDriver::SerialPortConfig cfg { /*baud*/ 115200,
                                                      /*csize*/ 8,
                                                      asio::serial_port_base::parity::none,
                                                      asio::serial_port_base::stop_bits::one,
                                                      asio::serial_port_base::flow_control::none };
    serial.init_port("", cfg);

    std::cout << "Serial port opened" << std::endl;
    serial.set_error_callback([&](const asio::error_code& ec) {
        std::cout << "serial error: " << ec.message() << std::endl;
    });
    serial.start();
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}