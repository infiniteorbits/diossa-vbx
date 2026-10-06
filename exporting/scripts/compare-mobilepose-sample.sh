#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

VENV_PATH="$REPO_ROOT_DIR/exporting/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH. Running setup_env.sh to create it..."
    bash $REPO_ROOT_DIR/exporting/setup_env.sh
fi
source $REPO_ROOT_DIR/exporting/.venv/bin/activate

python \
    -m diossa_model_exporter.compare-sample \
    --sample-dir "$REPO_ROOT_DIR/output/images/ccn1-kr-224x224-test-sample" \
    --task keypoints \
    --input-width 224 \
    --input-height 224 \
    --heatmap-width 56 \
    --heatmap-height 56 \
    "$@"
