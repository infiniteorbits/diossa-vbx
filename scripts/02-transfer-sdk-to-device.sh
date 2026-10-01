#!/bin/bash

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEVICE_IP="${DEVICE_IP:-192.168.20.6}"
DEVICE_USERNAME="${DEVICE_USERNAME:-root}"

function transfer_sdk_to_device() {
    local sdk_path="$1"
    local device_path="$2"

    ssh $DEVICE_USERNAME@$DEVICE_IP "mkdir -p $device_path"
    rsync -avzP \
        "${sdk_path}/drivers" \
        "${sdk_path}/example" \
        "${sdk_path}/lib" \
        "$DEVICE_USERNAME@$DEVICE_IP:$device_path"
}

transfer_sdk_to_device "${REPO_ROOT_DIR}/third-party/vbx-sdk" "/root/vbx-sdk"