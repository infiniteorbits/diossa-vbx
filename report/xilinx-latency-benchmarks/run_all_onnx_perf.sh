#!/bin/bash

model_dir=$1

find $model_dir -name "*.onnx" | while read -r onnx_model; do
    echo "Processing: $onnx_model"
    # Put your perf_test and parsing logic here
    bash run_single_onnx_perf.sh $onnx_model "results_onnx_${model_dir}.txt"
done
