#!/bin/bash

set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

(
    MODEL_CKPT="${MODEL_CKPT:-data/models/mobilepose/ccn1--blest-harl-epoch138.ckpt}"
    MODEL_INPUT_BCHW_SHAPE="${MODEL_INPUT_BCHW_SHAPE:-1 3 224 224}"
    MODEL_CLASS="${MODEL_CLASS:-nn_models.pytorch.keypoints_regression.mobilepose.MobilePose}"
    MODEL_CLASS_POSTPROCESSING="${MODEL_CLASS_POSTPROCESSING:-nn_models.pytorch.keypoints_regression.mobilepose.MobilePose_Edge}"

    CALIBRATION_IMAGES_DIR="${CALIBRATION_IMAGES_DIR:-data/images/pco-vis-test-sample/}"

    EXPORTED_MODEL_OUTPUT_DIR=$(basename ${MODEL_CKPT/.ckpt/})
    OUTPUT_DIR="${OUTPUT_DIR:-output/embedded-models/${EXPORTED_MODEL_OUTPUT_DIR}/}"

    bash $REPO_ROOT_DIR/exporting/scripts/export-pytorch-model.sh

    echo "===> Mobilepose model exported to ${REPO_ROOT_DIR}/${OUTPUT_DIR}"
)

