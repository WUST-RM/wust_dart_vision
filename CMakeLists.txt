cmake_minimum_required(VERSION 3.10)
project(dart_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ARM_BUILD "Build for ARM" OFF)
option(X86_BUILD "Build for X86" OFF)

if(ARM_BUILD)
    add_compile_definitions(ARM)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/dart_vision_ARM)
    set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

    set(OpenCV_DIR /home/hy/RV1106/opencv-install/opt/opencv-armhf/lib/cmake/opencv4)
    
    set(JPEG_TURBO_ROOT /home/hy/RV1106/libjpeg-turbo-install)

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

    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    add_compile_options(${ARM_FLAGS})
    add_link_options(${ARM_FLAGS})
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath,'$ORIGIN/../lib' -Wl,--as-needed")
    link_directories(${JPEG_TURBO_ROOT}/lib)
elseif(X86_BUILD)
    add_compile_definitions(X86)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/dart_vision_X86)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    set(JPEG_TURBO_ROOT /opt/libjpeg-turbo)
    link_directories(${JPEG_TURBO_ROOT}/lib64)
endif()

set(CMAKE_BUILD_TYPE Release)


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/3rdparty/tomlplusplus/include
    ${PROJECT_SOURCE_DIR}/3rdparty/asio/include
    ${JPEG_TURBO_ROOT}/include
)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} turbojpeg Eigen3::Eigen)


add_executable(vision src/dart_vision.cpp)
target_link_libraries(vision  ${PROJECT_NAME})

