#pragma once
#include <chrono>

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