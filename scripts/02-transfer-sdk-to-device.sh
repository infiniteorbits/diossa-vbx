#!/bin/bash

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEVICE_IP="${DEVICE_IP:-192.168.20.6}"
DEVICE_PORT="${DEVICE_PORT:-22}"
DEVICE_USERNAME="${DEVICE_USERNAME:-root}"
DEVICE_SDK_PATH="${DEVICE_SDK_PATH:-/root/vbx-sdk}"

function transfer_sdk_to_device() {
    local sdk_path="$1"
    local device_sdk_path="$2"

    ssh -p $DEVICE_PORT $DEVICE_USERNAME@$DEVICE_IP "mkdir -p $device_sdk_path"
    rsync -avzP \
        -e "ssh -p $DEVICE_PORT" \
        "${sdk_path}/drivers" \
        "${sdk_path}/apps" \
        "${sdk_path}/lib" \
        "$DEVICE_USERNAME@$DEVICE_IP:$device_sdk_path"
}

transfer_sdk_to_device "${REPO_ROOT_DIR}/third-party/vbx-sdk" "${DEVICE_SDK_PATH}"