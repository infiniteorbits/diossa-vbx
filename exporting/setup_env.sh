#!/bin/bash

set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$(dirname "$0")"

echo "===> Setting up model exporting environment..."
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: 'uv' is not installed. Please install 'uv' before running this script."
    exit 1
fi

if ! uv python find 3.12 >/dev/null 2>&1; then
    echo "Python 3.12 is not installed in uv. Installing Python 3.12 via uv..."
    uv python install 3.12
else
    echo "Python 3.12 is already available in uv."
fi

VENV_DIR="$REPO_ROOT_DIR/exporting/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR using Python 3.12..."
    uv venv "$VENV_DIR" --python=3.12
else
    echo "Virtual environment already exists in $VENV_DIR."
fi


source $VENV_DIR/bin/activate

uv pip install -e $REPO_ROOT_DIR/exporting/.
deactivate

echo "===> Model exporting environment setup complete."