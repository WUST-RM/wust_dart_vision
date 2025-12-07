# 目标平台
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# 交叉编译器
set(CMAKE_SYSROOT /opt/arm/sysroot)
set(CMAKE_C_COMPILER   /usr/bin/arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabihf-g++)

set(CMAKE_C_COMPILER_TARGET arm-linux-gnueabihf)
set(CMAKE_CXX_COMPILER_TARGET arm-linux-gnueabihf)

set(CMAKE_C_FLAGS   "--sysroot=${CMAKE_SYSROOT}")
set(CMAKE_CXX_FLAGS "--sysroot=${CMAKE_SYSROOT}")

set(CMAKE_EXE_LINKER_FLAGS "--sysroot=${CMAKE_SYSROOT} \
    -Wl,-rpath-link,${CMAKE_SYSROOT}/usr/lib/arm-linux-gnueabihf \
    -Wl,-rpath-link,${CMAKE_SYSROOT}/lib/arm-linux-gnueabihf \
    -L${CMAKE_SYSROOT}/usr/lib/arm-linux-gnueabihf \
    -L${CMAKE_SYSROOT}/lib/arm-linux-gnueabihf"
)

set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
