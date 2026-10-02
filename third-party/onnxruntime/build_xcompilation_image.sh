#!/bin/bash
set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker build \
    -t ort-riscv-builder \
    -f ${REPO_ROOT_DIR}/third-party/onnxruntime/Dockerfile \
    ${REPO_ROOT_DIR}