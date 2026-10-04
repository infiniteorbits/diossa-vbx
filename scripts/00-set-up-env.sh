#!/bin/bash

set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "===> Setting up VectorBlox SDK environment..."
cd $REPO_ROOT_DIR/third-party/vbx-sdk

echo "=====> Installing VectorBlox SDK dependencies..."
bash install_dependencies.sh

echo "=====> Removing existing VectorBlox SDK environment..."
rm -rf vbx_env

echo "=====> Setting up VectorBlox SDK environment..."
source setup_vars.sh
deactivate

cd $REPO_ROOT_DIR

echo "===> Setting up model exporting environment..."
bash $REPO_ROOT_DIR/exporting/setup_env.sh

echo "===> Done."
