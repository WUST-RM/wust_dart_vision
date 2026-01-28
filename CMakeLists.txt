cmake_minimum_required(VERSION 3.10)

project(dart_vision LANGUAGES CXX)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


option(ARM_BUILD "Build for ARM" OFF)
option(X86_BUILD "Build for X86" OFF)
if(ARM_BUILD)
    add_compile_definitions(ARM)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/ARM)
    set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)
endif()
if(X86_BUILD)
    add_compile_definitions(X86)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/X86)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()
find_package(Eigen3 REQUIRED)
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/src
${PROJECT_SOURCE_DIR}/3rdparty/tomlplusplus/include
${PROJECT_SOURCE_DIR}/3rdparty/asio/include)
add_subdirectory(3rdparty/tomlplusplus)

add_executable(camera test/camera.cpp)
target_link_libraries(camera PRIVATE Eigen3::Eigen)

add_executable(eigen_test test/eigen.cpp)
target_link_libraries(eigen_test PRIVATE Eigen3::Eigen)

add_executable(toml_test test/toml.cpp)
target_link_libraries(toml_test PRIVATE tomlplusplus::tomlplusplus)

add_executable(asio_test test/asio.cpp)
