#!/bin/bash
set -e

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SYSROOT_DIR="$REPO_ROOT_DIR/output/sysroot"
OUTPUT_DIR="$REPO_ROOT_DIR/output/onnxruntime-riscv64"

mkdir -p "$OUTPUT_DIR/raw"

docker run --rm -it \
  -v "$SYSROOT_DIR:/sysroot" \
  -v "$OUTPUT_DIR:/output" \
  ort-riscv-builder bash -c '
    set -e

    PYTHON_BIN="/opt/venv/bin/python"

    # Remove all sysroot Protobuf headers to prevent include collisions
    rm -rf /sysroot/usr/include/google/protobuf
    rm -rf /sysroot/usr/include/riscv64-linux-gnu/google/protobuf
    rm -rf /sysroot/usr/local/include/google/protobuf

    # Provide missing Linux 6.4+ RISC-V hwprobe header and syscall number for Ubuntu 22.04 sysroot
    mkdir -p /sysroot/usr/include/asm
    echo "===> Creating stub asm/hwprobe.h for Linux 5.15 sysroot..."
    cat << "EOF" > /sysroot/usr/include/asm/hwprobe.h
#ifndef _ASM_HWPROBE_H
#define _ASM_HWPROBE_H

#include <stdint.h>

#ifndef __NR_riscv_hwprobe
#define __NR_riscv_hwprobe 258
#endif

struct riscv_hwprobe {
    int64_t key;
    uint64_t value;
};

#define RISCV_HWPROBE_KEY_MVENDORID 0
#define RISCV_HWPROBE_KEY_MARCHID 1
#define RISCV_HWPROBE_KEY_MIMPID 2
#define RISCV_HWPROBE_KEY_BASE_BEHAVIOR 3
#define RISCV_HWPROBE_BASE_BEHAVIOR_IMA (1 << 0)
#define RISCV_HWPROBE_KEY_IMA_EXT_0 4
#define RISCV_HWPROBE_IMA_FD (1 << 0)
#define RISCV_HWPROBE_IMA_C (1 << 1)
#define RISCV_HWPROBE_IMA_V (1 << 2)

#endif
EOF

    # Dynamically match host protoc version to ONNX Runtime cmake/deps.txt
    PROTOBUF_DEP=$(grep -E "^protobuf;" /onnxruntime/cmake/deps.txt || true)
    PROTOBUF_VER=$(echo "$PROTOBUF_DEP" | sed -E "s/.*\/v?([0-9]+\.[0-9]+(\.[0-9]+)?).*/\1/")

    if [ -z "$PROTOBUF_VER" ]; then
      PROTOBUF_VER="25.1"
    fi

    echo "===> Detected ONNX Runtime Protobuf version: $PROTOBUF_VER"
    mkdir -p /opt/protoc

    if [ ! -f "/opt/protoc/bin/protoc" ] || ! /opt/protoc/bin/protoc --version | grep -q "$PROTOBUF_VER"; then
      echo "===> Installing matching host protoc compiler..."
      rm -rf /opt/protoc/*
      curl -LsSf "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VER}/protoc-${PROTOBUF_VER}-linux-x86_64.zip" -o /tmp/protoc.zip || \
      curl -LsSf "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VER}/protoc-3.${PROTOBUF_VER}-linux-x86_64.zip" -o /tmp/protoc.zip
      unzip -q -o /tmp/protoc.zip -d /opt/protoc
      rm -f /tmp/protoc.zip
      chmod +x /opt/protoc/bin/protoc
    fi

    # Wipe prior build directory inside container for a clean build
    rm -rf build/riscv64

    echo "===> Running ONNX Runtime C/C++ Library Build..."
    $PYTHON_BIN tools/ci_build/build.py \
      --allow_running_as_root \
      --config Release \
      --build_dir build/riscv64 \
      --build_shared_lib \
      --parallel \
      --skip_tests \
      --rv64 \
      --riscv_toolchain_root=/usr \
      --riscv_qemu_path=/usr/bin/qemu-riscv64-static \
      --use_xnnpack \
      --cmake_extra_defines \
        onnxruntime_DEV_MODE=OFF \
        FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER \
        XNNPACK_ENABLE_RISCV_VECTOR=OFF \
        CMAKE_C_FLAGS="--sysroot=/sysroot" \
        CMAKE_CXX_FLAGS="--sysroot=/sysroot -Wno-error=maybe-uninitialized -Wno-error=array-bounds -Wno-error=stringop-overflow -Wno-array-bounds -Wno-stringop-overflow" \
        CMAKE_EXE_LINKER_FLAGS="--sysroot=/sysroot" \
        CMAKE_SHARED_LINKER_FLAGS="--sysroot=/sysroot" \
        CMAKE_C_STANDARD_LIBRARIES="-latomic" \
        CMAKE_CXX_STANDARD_LIBRARIES="-latomic" \
        CMAKE_SYSROOT=/sysroot \
        CMAKE_FIND_ROOT_PATH="/sysroot" \
        ONNX_CUSTOM_PROTOC_EXECUTABLE=/opt/protoc/bin/protoc \
        FLATBUFFERS_FLATC_EXECUTABLE=$(which flatc)

    echo "===> Copying raw shared library to /output/raw/"
    rm -rf /output/raw/*
    cp -d build/riscv64/Release/libonnxruntime.so* /output/raw/

    echo "===> Shared library build complete! Saved to $OUTPUT_DIR/raw/"
'
