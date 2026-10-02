#!/bin/bash

set -euo pipefail

DEVICE_IP="${DEVICE_IP:-192.168.20.6}"
DEVICE_USERNAME="${DEVICE_USERNAME:-root}"
DEVICE_ONNXRUNTIME_PATH="${DEVICE_ONNXRUNTIME_PATH:-/root/vbx-sdk/onnxruntime-riscv64}"

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

echo "===> Transferring onnxruntime-riscv64 library and headers to device..."
rsync -avz "${REPO_ROOT_DIR}/output/onnxruntime-riscv64/" "${DEVICE_USERNAME}@${DEVICE_IP}:${DEVICE_ONNXRUNTIME_PATH}"
