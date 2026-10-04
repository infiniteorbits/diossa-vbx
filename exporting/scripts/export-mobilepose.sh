#!/bin/bash

set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

(
    export MODEL_CKPT="data/models/mobilepose/ccn1--blest-harl-epoch138.ckpt"
    export MODEL_INPUT_BCHW_SHAPE="1 3 224 224"
    export MODEL_CLASS="nn_models.pytorch.keypoints_regression.mobilepose.MobilePose"
    export MODEL_CLASS_POSTPROCESSING="nn_models.pytorch.keypoints_regression.mobilepose.MobilePose_Edge"
    export MODEL_KWARGS="keypoints=20 channels=3 ignore_invisible_keypoints=false normalized_coordinates=false"


    export MODEL_BACKBONE_OUTPUT_LAYER_NAMES="unnormalized_heatmaps"
    export MODEL_POSTPROCESSING_OUTPUT_LAYER_NAMES="coords heatmaps"


    export CALIBRATION_IMAGES_DIR="output/images/ccn1-kr-224x224-test-sample/raw"

    export EXPORTED_MODEL_OUTPUT_DIR="$(basename ${MODEL_CKPT/.ckpt/})"
    export OUTPUT_DIR="output/embedded-models/${EXPORTED_MODEL_OUTPUT_DIR}/"

    bash $REPO_ROOT_DIR/exporting/scripts/export-pytorch-model.sh

    echo "===> Mobilepose model exported to ${REPO_ROOT_DIR}/${OUTPUT_DIR}"
)

