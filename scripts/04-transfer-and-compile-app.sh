#!/bin/bash

set -euo pipefail

DEVICE_IP="${DEVICE_IP:-192.168.20.6}"
DEVICE_PORT="${DEVICE_PORT:-22}"
DEVICE_USERNAME="${DEVICE_USERNAME:-root}"
DEVICE_SDK_PATH="${DEVICE_SDK_PATH:-/root/vbx-sdk}"

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

APP_NAME="${APP_NAME:-single-model-runner}"

TEST_MODEL_VNNX=models/mobilepose_V1000_ncomp.vnnx
TEST_MODEL_ONNX=models/mobilepose-post-processing.onnx
TEST_MODEL_IMAGE=models/sample_images_224x224/eutelsat-00040.jpg


echo "===> Transferring apps to device..."
rsync -avzP \
    -e "ssh -p $DEVICE_PORT" \
    "${REPO_ROOT_DIR}/apps/${APP_NAME}" \
    "${DEVICE_USERNAME}@${DEVICE_IP}:${DEVICE_SDK_PATH}/apps/"

echo "===> Compiling app on device..."
ssh \
    -p $DEVICE_PORT \
    $DEVICE_USERNAME@$DEVICE_IP \
    "cd $DEVICE_SDK_PATH/apps/${APP_NAME} && (make overlay; make; make stage)"

echo "===> Transferring models/ to device..."
ssh -p $DEVICE_PORT $DEVICE_USERNAME@$DEVICE_IP "mkdir -p $DEVICE_SDK_PATH/models/"
rsync -avzP \
    -e "ssh -p $DEVICE_PORT" \
    "${REPO_ROOT_DIR}/${TEST_MODEL_VNNX}" \
    "${REPO_ROOT_DIR}/${TEST_MODEL_ONNX}" \
    "${REPO_ROOT_DIR}/${TEST_MODEL_IMAGE}" \
    "${DEVICE_USERNAME}@${DEVICE_IP}:${DEVICE_SDK_PATH}/models/"

echo "====> Test application on device..."
ssh \
    -p $DEVICE_PORT \
    $DEVICE_USERNAME@$DEVICE_IP \
    "cd $DEVICE_SDK_PATH/apps/${APP_NAME} && ./run-model ${TEST_MODEL_VNNX} ${TEST_MODEL_IMAGE} ${TEST_MODEL_ONNX}"