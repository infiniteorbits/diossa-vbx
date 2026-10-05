#!/bin/bash

# You must have access to the registry.gitlab.com/lmo-space/ios/development/diossa-ccn/docs:latest image.

# Build the documentation for the PolarFire Xilinx Tradeoff project

REPO_ROOT_DIR=$(git rev-parse --show-toplevel)
cd $REPO_ROOT_DIR/reports
PROJECT="PolarFire_Xilinx_Tradeoff"

docker run \
    --rm \
    -v $REPO_ROOT_DIR/reports:/usr/local/project \
    --env PROJECT=$PROJECT \
    registry.gitlab.com/lmo-space/ios/development/diossa-ccn/docs:latest \
    make clean simplepdf             

cp \
    "${REPO_ROOT_DIR}/reports/_build/${PROJECT}/simplepdf/PL24-0001-PXT-0001 i1.0 - PolarFire Xilinx Tradeoff.pdf" \
    "${REPO_ROOT_DIR}/reports/${PROJECT}/${PROJECT}.pdf"