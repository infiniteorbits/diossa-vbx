#!/bin/bash

set -euo pipefail

MODEL_CKPT="${MODEL_CKPT:-data/models/mobilepose/ccn1--blest-harl-epoch138.ckpt}"
MODEL_INPUT_BCHW_SHAPE="${MODEL_INPUT_BCHW_SHAPE:-1 3 224 224}"
MODEL_CLASS="${MODEL_CLASS:-nn_models.pytorch.keypoints_regression.mobilepose.MobilePose}"
MODEL_CLASS_POSTPROCESSING="${MODEL_CLASS_POSTPROCESSING:-nn_models.pytorch.keypoints_regression.mobilepose.MobilePose_Edge}"

CALIBRATION_IMAGES_DIR="${CALIBRATION_IMAGES_DIR:-data/images/pco-vis-test-sample/}"

EXPORTED_MODEL_OUTPUT_DIR=$(basename ${MODEL_CKPT/.ckpt/})
OUTPUT_DIR="${OUTPUT_DIR:-output/embedded-models/${EXPORTED_MODEL_OUTPUT_DIR}/}"


echo "===> Exporting mobilepose model to ${OUTPUT_DIR}..."
mkdir -p ${OUTPUT_DIR}

echo "===> Exporting mobilepose model to ONNX..."

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

VENV_PATH="$REPO_ROOT_DIR/exporting/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH. Running setup_env.sh to create it..."
    bash $REPO_ROOT_DIR/exporting/setup_env.sh
fi

source $REPO_ROOT_DIR/exporting/.venv/bin/activate

MODEL_ONNX="${MODEL_ONNX:-${REPO_ROOT_DIR}/${OUTPUT_DIR}${EXPORTED_MODEL_OUTPUT_DIR}.onnx}"

if [ ! -f ${MODEL_ONNX} ]; then
    echo "===> Exporting mobilepose PyTorch model edge part to ONNX ${MODEL_ONNX}..."
    python -m diossa-model-exporter.export_pytorch_to_onnx \
        --ckpt ${REPO_ROOT_DIR}/${MODEL_CKPT} \
        --onnx ${MODEL_ONNX} \
        --input-shape ${MODEL_INPUT_BCHW_SHAPE} \
        --model-class ${MODEL_CLASS} \
        --input-names input \
        --output-names unnormalized_heatmaps
    echo "===> Mobilepose model edge part exported to ONNX."
else
    echo "===> Mobilepose model edge part already exported to ONNX ${MODEL_ONNX}."
fi

MODEL_ONNX_POSTPROCESSING="${MODEL_ONNX_POSTPROCESSING:-${REPO_ROOT_DIR}/${OUTPUT_DIR}${EXPORTED_MODEL_OUTPUT_DIR}_postprocessing.onnx}"

if [ ! -f ${MODEL_ONNX_POSTPROCESSING} ]; then
    echo "===> Exporting mobilepose PyTorch model postprocessing part to ONNX ${MODEL_ONNX_POSTPROCESSING}..."
    python -m diossa-model-exporter.export_pytorch_to_onnx \
        --onnx ${MODEL_ONNX_POSTPROCESSING} \
        --input-shape ${MODEL_INPUT_BCHW_SHAPE} \
        --model-class ${MODEL_CLASS_POSTPROCESSING} \
        --input-names input \
        --output-names unnormalized_heatmaps
    echo "===> Mobilepose model postprocessing part exported to ONNX."
else
    echo "===> Mobilepose model postprocessing part already exported to ONNX ${MODEL_ONNX_POSTPROCESSING}."
fi

echo "===> Activating VectorBlox SDK environment..."
source $REPO_ROOT_DIR/third-party/vbx-sdk/setup_vars.sh


echo "===> Renaming ONNX layers..."
sor4onnx \
    --input_onnx_file_path ${MODEL_ONNX} \
    --old_new 'input' 'images' \
    --mode full \
    --search_mode prefix_match \
    --output_onnx_file_path ${MODEL_ONNX}



echo "Checking for Numpy calibration data file..."

CALIBRATION_NUMPY_ARRAY=${REPO_ROOT_DIR}/${OUTPUT_DIR}/${EXPORTED_MODEL_OUTPUT_DIR}_calibration.npy

MODEL_INPUT_HEIGHT=$(echo $MODEL_INPUT_BCHW_SHAPE | cut -d' ' -f3)
MODEL_INPUT_WIDTH=$(echo $MODEL_INPUT_BCHW_SHAPE | cut -d' ' -f4)

if [ ! -f $CALIBRATION_NUMPY_ARRAY ]; then
    echo "Generating Numpy calibration data file with height ${MODEL_INPUT_HEIGHT} and width ${MODEL_INPUT_WIDTH}..."
    generate_npy \
        ${REPO_ROOT_DIR}/${CALIBRATION_IMAGES_DIR} \
        -o $CALIBRATION_NUMPY_ARRAY \
        -s $MODEL_INPUT_WIDTH $MODEL_INPUT_HEIGHT --norm 
else
    echo "===> Numpy calibration data file already generated at ${CALIBRATION_NUMPY_ARRAY}."
fi

MODEL_TFLITE="${MODEL_TFLITE:-${REPO_ROOT_DIR}/${OUTPUT_DIR}${EXPORTED_MODEL_OUTPUT_DIR}.tflite}"

if [ ! -f $MODEL_TFLITE ]; then
    echo "===> Running ONNX2TF..."

    
    mkdir -p ${OUTPUT_DIR}/saved_model
    cd ${OUTPUT_DIR}

    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi

    ls -la $CALIBRATION_NUMPY_ARRAY
    onnx2tf \
        -cind images $CALIBRATION_NUMPY_ARRAY [[[[0.5,0.5,0.5]]]] [[[[0.5,0.5,0.5]]]] \
        -ois images:1,3,${MODEL_INPUT_WIDTH},${MODEL_INPUT_HEIGHT} \
        -i ${MODEL_ONNX} \
        --output_signaturedefs \
        --output_integer_quantized_tflite
    cp saved_model/${EXPORTED_MODEL_OUTPUT_DIR}_full_integer_quant.tflite ${MODEL_TFLITE}
else
    echo "===> Model TFLite already generated at ${MODEL_TFLITE}."
fi

echo "Model TFLite generated."


MODEL_PREPROCESSED_TFLITE="${MODEL_PREPROCESSED_TFLITE:-${REPO_ROOT_DIR}/${OUTPUT_DIR}${EXPORTED_MODEL_OUTPUT_DIR}.pre.tflite}"

if [ ! -f $MODEL_PREPROCESSED_TFLITE ]; then
    echo "===> Preprocessing Model TFLite..."
    tflite_preprocess $MODEL_TFLITE  --mean 127.5 127.5 127.5 --scale 127.5 127.5 127.5
    echo "===> Model TFLite preprocessed."
else
    echo "===> Model TFLite already preprocessed at ${MODEL_PREPROCESSED_TFLITE}."
fi

MODEL_VNNX="${MODEL_VNNX:-${REPO_ROOT_DIR}/${OUTPUT_DIR}${EXPORTED_MODEL_OUTPUT_DIR}.vnnx}"
if [ ! -f $MODEL_VNNX ]; then
    echo "===> Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile \
        -s V1000 \
        -c ncomp \
        -t $MODEL_PREPROCESSED_TFLITE \
        -o ${MODEL_VNNX}
    echo "===> VNNX generated at ${MODEL_VNNX}."
else
    echo "===> VNNX already generated at ${MODEL_VNNX}."
fi

echo "===> Deactivating VectorBlox SDK environment..."
deactivate

echo "===> Done."
