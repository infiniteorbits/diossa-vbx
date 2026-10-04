#!/bin/bash

set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

(
    export MODEL_CKPT="data/models/fcos/ccn1--licit-weal-epoch75.ckpt"
    export MODEL_INPUT_BCHW_SHAPE="1 3 288 384"
    export MODEL_CLASS="nn_models.pytorch.object_detection.fcos.FCOS"
    export MODEL_CLASS_POSTPROCESSING="nn_models.pytorch.object_detection.fcos.FCOS_Edge"
    export MODEL_KWARGS="img_size=[384,288]"

    export MODEL_BACKBONE_OUTPUT_LAYER_NAMES=""
    export MODEL_POSTPROCESSING_OUTPUT_LAYER_NAMES="boxes scores labels"


    export CALIBRATION_IMAGES_DIR="output/images/ccn1-od-384x288-test-sample/raw"

    export EXPORTED_MODEL_OUTPUT_DIR="$(basename ${MODEL_CKPT/.ckpt/})"
    export OUTPUT_DIR="output/embedded-models/${EXPORTED_MODEL_OUTPUT_DIR}/"

    bash $REPO_ROOT_DIR/exporting/scripts/export-pytorch-model.sh

    echo "===> Mobilepose model exported to ${REPO_ROOT_DIR}/${OUTPUT_DIR}"
)

