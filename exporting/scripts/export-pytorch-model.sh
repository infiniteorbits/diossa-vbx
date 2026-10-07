#!/bin/bash

set -euo pipefail

# Exit if any required params are not set
if [ -z "${MODEL_CKPT:-}" ]; then
    echo "Error: MODEL_CKPT environment variable must be set." >&2
    exit 1
fi
if [ -z "${MODEL_INPUT_BCHW_SHAPE:-}" ]; then
    echo "Error: MODEL_INPUT_BCHW_SHAPE environment variable must be set." >&2
    exit 1
fi
if [ -z "${MODEL_CLASS:-}" ]; then
    echo "Error: MODEL_CLASS environment variable must be set." >&2
    exit 1
fi


if [ -z "${MODEL_CLASS_POSTPROCESSING:-}" ]; then
    echo "Error: MODEL_CLASS_POSTPROCESSING environment variable must be set." >&2
    exit 1
fi

if [ -z "${CALIBRATION_IMAGES_DIR:-}" ]; then
    echo "Error: CALIBRATION_IMAGES_DIR environment variable must be set." >&2
    exit 1
fi

if [ -z "${NORMALIZATION_MEAN_FLOAT32:-}" ]; then
    echo "Error: NORMALIZATION_MEAN_FLOAT32 environment variable must be set." >&2
    exit 1
fi
if [ -z "${NORMALIZATION_STD_FLOAT32:-}" ]; then
    echo "Error: NORMALIZATION_STD_FLOAT32 environment variable must be set." >&2
    exit 1
fi

if [ -z "${REMAPPER_CLASS:-}" ]; then
    echo "Error: REMAPPER_CLASS environment variable must be set." >&2
    exit 1
fi

MODEL_KWARGS="${MODEL_KWARGS:-}"

MODEL_BACKBONE_OUTPUT_LAYER_NAMES="${MODEL_BACKBONE_OUTPUT_LAYER_NAMES:-}"
MODEL_POSTPROCESSING_OUTPUT_LAYER_NAMES="${MODEL_POSTPROCESSING_OUTPUT_LAYER_NAMES:-}"


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
MODEL_ONNX_POSTPROCESSING="${MODEL_ONNX_POSTPROCESSING:-${REPO_ROOT_DIR}/${OUTPUT_DIR}${EXPORTED_MODEL_OUTPUT_DIR}_postprocessing.onnx}"

NEED_POSTPROCESSING=0
if [ -n "${MODEL_CLASS_POSTPROCESSING}" ] && [ ! -f "${MODEL_ONNX_POSTPROCESSING}" ]; then
    NEED_POSTPROCESSING=1
fi

if [ ! -f "${MODEL_ONNX}" ] || [ "${NEED_POSTPROCESSING}" -eq 1 ]; then
    echo "===> Exporting PyTorch model to ONNX ${MODEL_ONNX}..."
    EXPORT_CMD=(
        python -m diossa_model_exporter.export_pytorch_to_onnx
        --ckpt "${REPO_ROOT_DIR}/${MODEL_CKPT}"
        --onnx "${MODEL_ONNX}"
        --input-shape ${MODEL_INPUT_BCHW_SHAPE}
        --model-class "${MODEL_CLASS}"
        --remapper-class "${REMAPPER_CLASS}"
        --input-names input
    )
    if [ -n "${MODEL_KWARGS}" ]; then
        # shellcheck disable=SC2206
        EXPORT_CMD+=(--model-kwargs ${MODEL_KWARGS})
    fi
    if [ -n "${MODEL_BACKBONE_OUTPUT_LAYER_NAMES}" ]; then
        # shellcheck disable=SC2206
        EXPORT_CMD+=(--output-names ${MODEL_BACKBONE_OUTPUT_LAYER_NAMES})
    fi
    if [ -n "${MODEL_CLASS_POSTPROCESSING}" ]; then
        EXPORT_CMD+=(
            --postprocessing-model-class "${MODEL_CLASS_POSTPROCESSING}"
            --postprocessing-onnx "${MODEL_ONNX_POSTPROCESSING}"
        )
        if [ -n "${MODEL_POSTPROCESSING_OUTPUT_LAYER_NAMES}" ]; then
            # shellcheck disable=SC2206
            EXPORT_CMD+=(--postprocessing-output-names ${MODEL_POSTPROCESSING_OUTPUT_LAYER_NAMES})
        fi
    fi
    echo ${EXPORT_CMD[@]}
    "${EXPORT_CMD[@]}"
    echo "===> PyTorch model exported to ONNX."
else
    echo "===> PyTorch model already exported to ONNX ${MODEL_ONNX}."
fi

echo "Checking for Numpy calibration data file..."

CALIBRATION_NUMPY_ARRAY=${REPO_ROOT_DIR}/${OUTPUT_DIR}/${EXPORTED_MODEL_OUTPUT_DIR}_calibration.npy

MODEL_INPUT_HEIGHT=$(echo $MODEL_INPUT_BCHW_SHAPE | cut -d' ' -f3)
MODEL_INPUT_WIDTH=$(echo $MODEL_INPUT_BCHW_SHAPE | cut -d' ' -f4)

if [ ! -f $CALIBRATION_NUMPY_ARRAY ]; then
    echo "Generating Numpy calibration data file with height ${MODEL_INPUT_HEIGHT} and width ${MODEL_INPUT_WIDTH}..."
    python -m diossa_model_exporter.make_calibration_sample \
        ${REPO_ROOT_DIR}/${CALIBRATION_IMAGES_DIR} \
        -o $CALIBRATION_NUMPY_ARRAY \
        --count 100 \
        -s $MODEL_INPUT_HEIGHT $MODEL_INPUT_WIDTH --norm
else
    echo "===> Numpy calibration data file already generated at ${CALIBRATION_NUMPY_ARRAY}."
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


MODEL_TFLITE="${MODEL_TFLITE:-${REPO_ROOT_DIR}/${OUTPUT_DIR}${EXPORTED_MODEL_OUTPUT_DIR}.tflite}"

if [ ! -f $MODEL_TFLITE ]; then
    echo "===> Running ONNX2TF..."

    
    mkdir -p ${OUTPUT_DIR}/saved_model
    cd ${OUTPUT_DIR}

    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi

    echo "===> Running ONNXSIM..."
    MODEL_ONNX_SIM="${MODEL_ONNX_SIM:-${REPO_ROOT_DIR}/${OUTPUT_DIR}${EXPORTED_MODEL_OUTPUT_DIR}_sim.onnx}"

    onnxsim ${MODEL_ONNX} ${MODEL_ONNX_SIM}
    mv ${MODEL_ONNX_SIM} ${MODEL_ONNX}

    # Mean/std must be NHWC with a leading batch axis: [[[[r,g,b]]]].
    # Three brackets broadcast to (H, W, 3). TFLite then resizes the graph
    # input to rank 3, and the leading Pad (paddings [4, 2]) fails to prepare.
    onnx2tf \
        -cind images "$CALIBRATION_NUMPY_ARRAY" "[[[[${NORMALIZATION_MEAN_FLOAT32}]]]]" "[[[[${NORMALIZATION_STD_FLOAT32}]]]]" \
        -ois images:1,3,${MODEL_INPUT_HEIGHT},${MODEL_INPUT_WIDTH} \
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

    # Comma-separated float values, e.g. "0.485,0.456,0.406", scaled into 0..255.
    NORMALIZATION_MEAN_AS_INT8_DECIMAL=$(python3 -c "print(' '.join(format(float(x) * 255, '.10g') for x in '${NORMALIZATION_MEAN_FLOAT32}'.replace(',', ' ').split()))")
    NORMALIZATION_STD_AS_INT8_DECIMAL=$(python3 -c "print(' '.join(format(float(x) * 255, '.10g') for x in '${NORMALIZATION_STD_FLOAT32}'.replace(',', ' ').split()))")

    tflite_preprocess $MODEL_TFLITE  --mean ${NORMALIZATION_MEAN_AS_INT8_DECIMAL} --scale ${NORMALIZATION_STD_AS_INT8_DECIMAL}
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


if [ -f ${MODEL_VNNX} ]; then
    echo "Running Simulation..."
    #python $VBX_SDK/example/python/ssdv2.py fcos_V1000_ncomp.vnnx images/PCO_VIS_EUTELSAT_v4_CCN2_val_sample_13_image.png --torch 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model fcos_V1000_ncomp.vnnx images/PCO_VIS_EUTELSAT_v4_CCN2_val_sample_13_image.png'
fi

echo "===> Deactivating VectorBlox SDK environment..."
deactivate

echo "===> Done."
