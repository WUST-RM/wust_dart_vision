cmake_minimum_required(VERSION 3.10)

project(dart_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ARM_BUILD "Build for ARM" OFF)
option(X86_BUILD "Build for X86" OFF)

if(ARM_BUILD)
    add_compile_definitions(ARM)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/ARM)
    set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

    # OpenCV 路径
    set(OpenCV_DIR /home/hy/RV1106/opencv-install/opt/opencv-armhf/lib/cmake/opencv4)

    # ARMv7 + NEON + 硬浮点
    set(ARM_FLAGS
        "-march=armv7-a"
        "-mfpu=neon-vfpv3"
        "-mfloat-abi=hard"
        "-O3"
        "-ffast-math"
        "-funroll-loops"
        "-ftree-vectorize"
        "-fomit-frame-pointer"
    )

    # LTO（链接期优化）
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

    add_compile_options(${ARM_FLAGS})
    add_link_options(${ARM_FLAGS})

    # rpath
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath,'$ORIGIN/../lib' -Wl,--as-needed")

endif()

if(X86_BUILD)
    add_compile_definitions(X86)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/X86)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/3rdparty/tomlplusplus/include
    ${PROJECT_SOURCE_DIR}/3rdparty/asio/include
)

add_subdirectory(3rdparty/tomlplusplus)
add_executable(vision src/dart_vision.cpp)
target_link_libraries(vision PRIVATE ${OpenCV_LIBS})

add_executable(camera test/camera.cpp)
target_link_libraries(camera PRIVATE ${OpenCV_LIBS})

add_executable(toml_test test/toml.cpp)
target_link_libraries(toml_test PRIVATE tomlplusplus::tomlplusplus)

add_executable(asio_test test/asio.cpp)