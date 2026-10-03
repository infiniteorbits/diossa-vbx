#!/bin/bash

set -euo pipefail

DEVICE_IP="${DEVICE_IP:-192.168.20.6}"
DEVICE_PORT="${DEVICE_PORT:-22}"
DEVICE_USERNAME="${DEVICE_USERNAME:-root}"
DEVICE_SDK_PATH="${DEVICE_SDK_PATH:-/root/vbx-sdk}"

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker &> /dev/null; then
  echo "Error: Docker is not installed. Please install Docker and try again." >&2
  exit 1
fi

if ! docker image inspect ort-riscv-builder > /dev/null 2>&1; then
  echo "===> ort-riscv-builder Docker image not found; building it now..."
  bash "${REPO_ROOT_DIR}/third-party/onnxruntime/build_xcompilation_image.sh"
fi

echo "===> Compiling ONNX Runtime for RISC-V..."
bash "${REPO_ROOT_DIR}/third-party/onnxruntime/xcompile_onnxruntime_without_python.sh"

echo "===> Patching ONNX Runtime library for RISC-V..."
bash "${REPO_ROOT_DIR}/third-party/onnxruntime/patch_libraries.sh"

echo "===> Transferring onnxruntime-riscv64 library and headers to device..."
rsync -avzP \
    -e "ssh -p $DEVICE_PORT" \
    "${REPO_ROOT_DIR}/output/onnxruntime-riscv64/include" \
    "${REPO_ROOT_DIR}/output/onnxruntime-riscv64/lib" \
    "${DEVICE_USERNAME}@${DEVICE_IP}:${DEVICE_SDK_PATH}/onnxruntime-riscv64"
