#define ASIO_STANDALONE
#include <asio.hpp>
#include <atomic>
#include <chrono>
#include <functional>
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
    using ReceiveCallback = std::function<void(const uint8_t*, std::size_t)>;
    using ErrorCallback = std::function<void(const asio::error_code&)>;
    using Ptr = std::shared_ptr<SerialDriver>;

    struct SerialPortConfig {
        unsigned baud_rate = 115200;
        unsigned char_size = 8;
        asio::serial_port_base::parity::type parity = asio::serial_port_base::parity::none;
        asio::serial_port_base::stop_bits::type stop_bits = asio::serial_port_base::stop_bits::one;
        asio::serial_port_base::flow_control::type flow_control =
            asio::serial_port_base::flow_control::none;
    };

    explicit SerialDriver(size_t read_buf_size = 4096):
        io_(),
        port_(io_),
        read_buf_(read_buf_size) {}
    static Ptr create() {
        return std::make_shared<SerialDriver>();
    }

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

    void start() {
        if (running_)
            return;
        running_ = true;
        worker_ = std::thread(&SerialDriver::run, this);
    }

    void stop() {
        running_ = false;
        close_port();
        if (worker_.joinable())
            worker_.join();
    }

    bool write(const std::vector<uint8_t>& data) {
        if (!port_.is_open())
            return false;
        asio::error_code ec;
        asio::write(port_, asio::buffer(data), ec);
        if (ec && error_cb_)
            error_cb_(ec);
        return !ec;
    }

    void run() {
        while (running_) {
            if (!open_port()) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            }
            read_loop();
            close_port();
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

        port_.set_option(asio::serial_port_base::baud_rate(config_.baud_rate));
        port_.set_option(asio::serial_port_base::character_size(config_.char_size));
        port_.set_option(asio::serial_port_base::parity(config_.parity));
        port_.set_option(asio::serial_port_base::stop_bits(config_.stop_bits));
        port_.set_option(asio::serial_port_base::flow_control(config_.flow_control));
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
            if (n && receive_cb_)
                receive_cb_(read_buf_.data(), n);
        }
    }

private:
    asio::io_context io_;
    asio::serial_port port_;
    std::vector<uint8_t> read_buf_;
    SerialPortConfig config_;
    std::string device_;

    std::atomic<bool> running_ { false };
    std::thread worker_;

    ReceiveCallback receive_cb_;
    ErrorCallback error_cb_;
};
} // namespace dart_vision
