cmake_minimum_required(VERSION 3.10)

set(CMAKE_TOOLCHAIN_FILE "/home/hy/RV1106/test/toolchain/toolchain-rv1106.cmake" CACHE PATH "Toolchain file")

set(CMAKE_BUILD_TYPE "Release")
project(dart_vision LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- 输出可执行文件到项目根目录 ----
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

# OpenCV
set(OpenCV_DIR "${CMAKE_SYSROOT}/usr/lib/arm-linux-gnueabihf/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS} ${yaml-cpp_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIR} )
list(APPEND BUILD_DEFINITIONS ARM)
include_directories(src/zcbor)
add_executable(dart_vision src/main.cpp src/zcbor/zcbor_common.c src/zcbor/zcbor_decode.c src/zcbor/zcbor_encode.c src/zcbor/zcbor_print.c)
target_compile_definitions(dart_vision PRIVATE ${BUILD_DEFINITIONS})
target_link_libraries(dart_vision
    PRIVATE
    ${OpenCV_LIBS}
    jpeg
    yaml-cpp
    yuv
    armadillo
    arpack
    superlu
    ${CMAKE_SYSROOT}/usr/lib/arm-linux-gnueabihf/openblas-pthread/liblapack.so
    ${CMAKE_SYSROOT}/usr/lib/arm-linux-gnueabihf/openblas-pthread/libopenblas.so
    m
    pthread
    dl
)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
    message(STATUS "Enable ARMv7 NEON optimization")
    target_compile_options(dart_vision PRIVATE -O3 -march=armv7-a -mfpu=neon -mfloat-abi=hard)
endif()







# X86

# cmake_minimum_required(VERSION 3.10)


# set(CMAKE_BUILD_TYPE "Release")
# project(dart_vision LANGUAGES C CXX)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# find_package(OpenCV REQUIRED)
# find_package(yaml-cpp REQUIRED)
# include_directories(${PROJECT_SOURCE_DIR})
# include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(src/zcbor)

# list(APPEND BUILD_DEFINITIONS X86)
# add_executable(dart_vision src/main.cpp src/zcbor/zcbor_common.c src/zcbor/zcbor_decode.c src/zcbor/zcbor_encode.c src/zcbor/zcbor_print.c)
# target_compile_definitions(dart_vision PRIVATE ${BUILD_DEFINITIONS})
# target_link_libraries(dart_vision
#     PRIVATE
#     ${OpenCV_LIBS}
#     jpeg
#     yaml-cpp
# )
