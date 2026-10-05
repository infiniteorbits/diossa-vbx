#!/bin/bash

set -euo pipefail

DEVICE_IP="${DEVICE_IP:-192.168.20.6}"
DEVICE_PORT="${DEVICE_PORT:-22}"
DEVICE_USERNAME="${DEVICE_USERNAME:-root}"
DEVICE_SDK_PATH="${DEVICE_SDK_PATH:-/root/vbx-sdk}"

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

APP_NAME="${APP_NAME:-single-model-runner}"

# Set to 1 to skip images whose JSON output is already on the device.
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# Keypoint regression (224x224).
MODEL_VNNX=output/embedded-models/ccn1--bijou-rasp-epoch17/ccn1--bijou-rasp-epoch17.vnnx
MODEL_ONNX=output/embedded-models/ccn1--bijou-rasp-epoch17/ccn1--bijou-rasp-epoch17_postprocessing.onnx
SAMPLE_DIR=output/images/ccn1-kr-224x224-test-sample

# Object detection (384x288).
# MODEL_VNNX=output/embedded-models/ccn1--quare-delf-epoch19/ccn1--quare-delf-epoch19.vnnx
# MODEL_ONNX=output/embedded-models/ccn1--quare-delf-epoch19/ccn1--quare-delf-epoch19_postprocessing.onnx
# SAMPLE_DIR=output/images/ccn1-od-384x288-test-sample

SAMPLE_RAW="${SAMPLE_DIR}/raw"
PRED_DIR="${SAMPLE_DIR}/pred"

shopt -s nullglob
images=( "${REPO_ROOT_DIR}/${SAMPLE_RAW}"/*.jpg )
shopt -u nullglob
if (( ${#images[@]} == 0 )); then
    echo "No JPEG images in ${SAMPLE_RAW}" >&2
    exit 1
fi

echo "===> Transferring app to device..."
rsync -avzP \
    -e "ssh -p $DEVICE_PORT" \
    "${REPO_ROOT_DIR}/apps/${APP_NAME}" \
    "${DEVICE_USERNAME}@${DEVICE_IP}:${DEVICE_SDK_PATH}/apps/"

echo "===> Compiling app on device..."
ssh \
    -p $DEVICE_PORT \
    $DEVICE_USERNAME@$DEVICE_IP \
    "cd $DEVICE_SDK_PATH/apps/${APP_NAME} && (make overlay; make; make stage)"

echo "===> Transferring model and ${#images[@]} sample images..."
ssh -p $DEVICE_PORT $DEVICE_USERNAME@$DEVICE_IP \
    "mkdir -p $DEVICE_SDK_PATH/${SAMPLE_RAW}"
rsync -avzP -R \
    -e "ssh -p $DEVICE_PORT" \
    "${REPO_ROOT_DIR}/./${MODEL_VNNX}" \
    "${REPO_ROOT_DIR}/./${MODEL_ONNX}" \
    "${DEVICE_USERNAME}@${DEVICE_IP}:${DEVICE_SDK_PATH}/"
rsync -avzP \
    -e "ssh -p $DEVICE_PORT" \
    "${REPO_ROOT_DIR}/${SAMPLE_RAW}/" \
    "${DEVICE_USERNAME}@${DEVICE_IP}:${DEVICE_SDK_PATH}/${SAMPLE_RAW}/"

echo "===> Running inference on ${#images[@]} images..."
ssh -p $DEVICE_PORT $DEVICE_USERNAME@$DEVICE_IP \
    sh -s -- \
    "$DEVICE_SDK_PATH" \
    "$SAMPLE_RAW" \
    "$MODEL_VNNX" \
    "$MODEL_ONNX" \
    "$APP_NAME" \
    "$SKIP_EXISTING" <<'EOF'
set -eu
sdk_path=$1
sample_raw=$2
model_vnnx=$3
model_onnx=$4
app_name=$5
skip_existing=$6

cd "$sdk_path"
ran=0
skipped=0
for img in "$sample_raw"/*.jpg; do
    if [ ! -f "$img" ]; then
        echo "No JPEG images in $sample_raw" >&2
        exit 1
    fi
    out="${img%.*}.json"
    if [ "$skip_existing" = 1 ] && [ -f "$out" ]; then
        echo "skip $img"
        skipped=$((skipped + 1))
        continue
    fi
    echo "===> $img"
    WRITE_OUT=1 "./apps/${app_name}/run-model" "$model_vnnx" "$img" "$model_onnx"
    ran=$((ran + 1))
done
echo "Ran $ran, skipped $skipped"
EOF

echo "===> Pulling predictions into ${PRED_DIR}..."
mkdir -p "${REPO_ROOT_DIR}/${PRED_DIR}"
rsync -avzP \
    -e "ssh -p $DEVICE_PORT" \
    --include='*.json' \
    --exclude='*' \
    "${DEVICE_USERNAME}@${DEVICE_IP}:${DEVICE_SDK_PATH}/${SAMPLE_RAW}/" \
    "${REPO_ROOT_DIR}/${PRED_DIR}/"

echo "===> Done. Predictions are in ${PRED_DIR}"
