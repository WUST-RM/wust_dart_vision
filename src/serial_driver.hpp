#pragma once
#include <asio.hpp>
#include <asio/read.hpp>
#include <asio/write.hpp>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
namespace dart_vision {
template<typename T>
inline T fromVector(const std::vector<uint8_t>& data) {
    T packet {};
    std::memcpy(&packet, data.data(), sizeof(T));
    return packet;
}

template<typename T>
inline std::vector<uint8_t> toVector(const T& data) {
    std::vector<uint8_t> packet(sizeof(T));
    std::memcpy(packet.data(), &data, sizeof(T));
    return packet;
}

class SerialDriver {
public:
    using ReceiveCallback = std::function<void(const uint8_t* data, std::size_t len)>;
    using ErrorCallback = std::function<void(const asio::error_code& ec)>;

    struct SerialPortConfig {
        unsigned int baud_rate = 115200;
        unsigned int char_size = 8;
        asio::serial_port_base::parity::type parity = asio::serial_port_base::parity::none;
        asio::serial_port_base::stop_bits::type stop_bits = asio::serial_port_base::stop_bits::one;
        asio::serial_port_base::flow_control::type flow_control =
            asio::serial_port_base::flow_control::none;
    };

    explicit SerialDriver(std::size_t read_buf_size = 4096):
        io_(),
        port_(io_),
        read_buf_(read_buf_size),
        running_(false) {}

    ~SerialDriver() {
        stop();
    }

    void init_port(const std::string& device, const SerialPortConfig& cfg) {
        device_ = device;
        config_ = cfg;
    }

    void set_receive_callback(ReceiveCallback cb) {
        receive_cb_ = std::move(cb);
    }
    void set_error_callback(ErrorCallback cb) {
        error_cb_ = std::move(cb);
    }

    bool start() {
        if (running_)
            return true;
        running_ = true;
        worker_thread_ = std::thread([this]() { this->run(); });
        return true;
    }

    void stop() {
        if (!running_)
            return;
        running_ = false;

        asio::error_code ec;
        if (port_.is_open()) {
            port_.cancel(ec);
            port_.close(ec);
        }

        if (worker_thread_.joinable())
            worker_thread_.join();
    }

    bool is_open() const {
        return port_.is_open();
    }

    bool write(const std::vector<uint8_t>& data) {
        if (!port_.is_open())
            return false;

        asio::error_code ec;
        size_t bytes_written = asio::write(port_, asio::buffer(data), ec);
        if (ec) {
            if (error_cb_)
                error_cb_(ec);
            return false;
        }
        return bytes_written == data.size();
    }
    bool write(const uint8_t* data, size_t size) {
        if (!port_.is_open() || data == nullptr || size == 0)
            return false;

        asio::error_code ec;
        size_t bytes_written = asio::write(port_, asio::buffer(data, size), ec);
        if (ec) {
            if (error_cb_)
                error_cb_(ec);
            return false;
        }
        return bytes_written == size;
    }

private:
    void run() {
        while (running_) {
            if (!open_port()) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            }

            read_loop();
            close_port();
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

    bool open_port() {
        asio::error_code ec;
        port_.open(device_, ec);
        if (ec) {
            if (error_cb_)
                error_cb_(ec);
            return false;
        }

        port_.set_option(asio::serial_port_base::baud_rate(config_.baud_rate), ec);
        port_.set_option(asio::serial_port_base::character_size(config_.char_size), ec);
        port_.set_option(asio::serial_port_base::parity(config_.parity), ec);
        port_.set_option(asio::serial_port_base::stop_bits(config_.stop_bits), ec);
        port_.set_option(asio::serial_port_base::flow_control(config_.flow_control), ec);

        return true;
    }

    void close_port() {
        asio::error_code ec;
        if (port_.is_open()) {
            port_.cancel(ec);
            port_.close(ec);
        }
    }

    void read_loop() {
        while (running_ && port_.is_open()) {
            asio::error_code ec;
            size_t n = port_.read_some(asio::buffer(read_buf_), ec);
            if (ec) {
                if (error_cb_)
                    error_cb_(ec);
                break;
            }
            if (n > 0 && receive_cb_) {
                receive_cb_(read_buf_.data(), n);
            }
        }
    }

private:
    asio::io_context io_;
    asio::serial_port port_;
    std::vector<uint8_t> read_buf_;
    SerialPortConfig config_;
    std::string device_;

    std::atomic<bool> running_;
    std::thread worker_thread_;

    ReceiveCallback receive_cb_;
    ErrorCallback error_cb_;
};
} // namespace dart_vision