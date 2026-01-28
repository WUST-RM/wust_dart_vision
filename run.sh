#!/usr/bin/env bash
set -e


WORK_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
BUILD_X86_DIR="$WORK_DIR/build-x86"
BUILD_ARM_DIR="$WORK_DIR/build-arm"
BUILD_FOR_IDE_DIR="$WORK_DIR/build"

ACTION="${1:-build}"   # 默认 build

echo "[INFO] Action: $ACTION"


if [[ "$ACTION" == "clean" || "$ACTION" == "rebuild" ]]; then
    echo "[INFO] Cleaning build directories..."
    rm -rf "$BUILD_X86_DIR" "$BUILD_ARM_DIR" "$BUILD_FOR_IDE_DIR"

    if [[ "$ACTION" == "clean" ]]; then
        echo "[INFO] Clean done."
        exit 0
    fi
fi


echo "[INFO] Building x86..."
cmake -S "$WORK_DIR" -B "$BUILD_FOR_IDE_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DX86_BUILD=ON

cmake -S "$WORK_DIR" -B "$BUILD_X86_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DX86_BUILD=ON

ninja -C "$BUILD_X86_DIR"


echo "[INFO] Building ARM..."
cmake -S "$WORK_DIR" -B "$BUILD_ARM_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DARM_BUILD=ON \
    -DCMAKE_TOOLCHAIN_FILE="$WORK_DIR/toolchain/toolchain-rv1106.cmake"

ninja -C "$BUILD_ARM_DIR"

echo "[INFO] Build finished successfully."