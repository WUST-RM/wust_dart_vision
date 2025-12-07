#include "serial_driver.hpp"
#include "zcbor_common.h"
#include "zcbor_decode.h"
#include "zcbor_encode.h"

namespace dart_vision {
struct SendPacket {
    float distance_x;
    float distance_y;
};
inline void toZcbor(dart_vision::SerialDriver& serial_driver, const SendPacket& packet) {
    uint8_t payload[128];
    ZCBOR_STATE_E(state, 2, payload, sizeof(payload), 1);

    zcbor_list_start_encode(state, 2);

    float distance_x = packet.distance_x;
    zcbor_float32_encode(state, &distance_x);
    float distance_y = packet.distance_y;
    zcbor_float32_encode(state, &distance_y);

    zcbor_list_end_encode(state, 2);

    serial_driver.write(payload, state->payload - payload);
}

} // namespace dart_vision
