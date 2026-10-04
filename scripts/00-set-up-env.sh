#!/bin/bash

set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash $REPO_ROOT_DIR/third-party/vbx-sdk/install_dependencies.sh
