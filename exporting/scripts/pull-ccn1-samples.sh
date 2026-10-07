#!/usr/bin/env bash

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"


VENV_PATH="$REPO_ROOT_DIR/exporting/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH. Running setup_env.sh to create it..."
    bash $REPO_ROOT_DIR/exporting/setup_env.sh
fi
source $REPO_ROOT_DIR/exporting/.venv/bin/activate

python \
    -m diossa_model_exporter.pull-dataset-sample \
    --output-dir $REPO_ROOT_DIR/output/images/ccn1-kr-224x224-test-sample/ \
    --dataset-name Inmarsat-5_DIOSSA-CCN_Pangu_v4 \
    --dataset-path DIOSSA_CCN1 \
    --split-name test \
    --branch main \
    --pipeline-name "[Keypoint Regression] AutoCrop -> Resize to 224x224" \
    --sample-count 100 \
    --seed 42 \
    --expected-keypoints 41 \
    --keypoint-indices 0 1 2 3 4 5 6 7 8 21 26 27 31 32 33 34 37 38 39 40


python \
    -m diossa_model_exporter.pull-dataset-sample \
    --output-dir $REPO_ROOT_DIR/output/images/ccn1-od-384x288-test-sample/ \
    --dataset-name Inmarsat-5_DIOSSA-CCN_Pangu_v4 \
    --dataset-path DIOSSA_CCN1 \
    --split-name test \
    --branch main \
    --pipeline-name "[Object Detection] CenterCrop -> Resize to 384x288" \
    --sample-count 100 \
    --seed 42 \
    --expected-keypoints 41 \
    --keypoint-indices 0 1 2 3 4 5 6 7 8 21 26 27 31 32 33 34 37 38 39 40