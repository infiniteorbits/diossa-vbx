#!/bin/bash

set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_comparison() {
    local label="$1"
    local sample_dir="$2"
    local script="$3"

    shopt -s nullglob
    local predictions=( "${sample_dir}/pred/"*.json )
    shopt -u nullglob
    if (( ${#predictions[@]} == 0 )); then
        echo "===> Skipping ${label}: no predictions in ${sample_dir}/pred"
        return 0
    fi

    echo "===> Comparing ${label} (${#predictions[@]} images)..."
    bash "$script"
}

run_comparison \
    "MobilePose keypoints" \
    "${REPO_ROOT_DIR}/output/images/ccn1-kr-224x224-test-sample" \
    "${REPO_ROOT_DIR}/exporting/scripts/compare-mobilepose-sample.sh"

run_comparison \
    "FCOS boxes" \
    "${REPO_ROOT_DIR}/output/images/ccn1-od-384x288-test-sample" \
    "${REPO_ROOT_DIR}/exporting/scripts/compare-fcos-sample.sh"
