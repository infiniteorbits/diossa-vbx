#!/usr/bin/env bash

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VENV_PATH="$REPO_ROOT_DIR/exporting/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH. Running setup_env.sh to create it..."
    bash $REPO_ROOT_DIR/exporting/setup_env.sh
fi
echo "===> Activating virtual environment..."
source $REPO_ROOT_DIR/exporting/.venv/bin/activate

echo "===> Pulling CCN1 samples..."
bash $REPO_ROOT_DIR/exporting/scripts/pull-ccn1-samples.sh

echo "===> Exporting Mobilepose model..."
bash $REPO_ROOT_DIR/exporting/scripts/export-mobilepose.sh

echo "===> Exporting FCOS model..."
bash $REPO_ROOT_DIR/exporting/scripts/export-fcos.sh