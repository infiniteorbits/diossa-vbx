#!/bin/bash

set -euo pipefail

# Measure VectorBlox and CPU post-processing latency on the PolarFire board.
#
# SSHes to the device, runs ./run-model on the first NUM_IMAGES JPEGs of each
# staged sample, greps the timing lines, writes a CSV, and summarizes it with
# diossa_model_exporter.summarize-latency into an RST list-table.
#
# Accelerator line:
#   network took 394.6820 ms (2.53 FPS) over 1 cycles
# CPU post-processing line:
#   dequantize took 1.7940 ms, ONNX Runtime took 13.4010 ms
#
# The single-model-runner binary must already be built on the device
# (scripts/05-transfer-compile-and-test-app.sh or scripts/06-run-inference-on-sample.sh).
#
#   NUM_IMAGES=8 ./scripts/08-measure-latency.sh

DEVICE_IP="${DEVICE_IP:-192.168.20.6}"
DEVICE_PORT="${DEVICE_PORT:-22}"
DEVICE_USERNAME="${DEVICE_USERNAME:-root}"
DEVICE_SDK_PATH="${DEVICE_SDK_PATH:-/root/vbx-sdk}"

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

APP_NAME="${APP_NAME:-single-model-runner}"
NUM_IMAGES="${NUM_IMAGES:-50}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT_DIR}/output/latency}"
CSV_PATH="${CSV_PATH:-${OUT_DIR}/latency.csv}"
RST_PATH="${RST_PATH:-${OUT_DIR}/latency.rst}"

if ! [[ "$NUM_IMAGES" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_IMAGES must be a positive integer (got '${NUM_IMAGES}')" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "model,image,network_ms,fps,loops,dequantize_ms,onnxruntime_ms" > "$CSV_PATH"

ssh_device() {
    ssh -p "$DEVICE_PORT" "${DEVICE_USERNAME}@${DEVICE_IP}" "$@"
}

echo "===> Checking run-model on ${DEVICE_USERNAME}@${DEVICE_IP}..."
ssh_device "test -x '${DEVICE_SDK_PATH}/apps/${APP_NAME}/run-model'"

parse_network_line() {
    local line="$1"
    if [[ ! "$line" =~ network\ took\ ([0-9]+\.[0-9]+)\ ms\ \(([0-9]+\.[0-9]+)\ FPS\)\ over\ ([0-9]+)\ cycles ]]; then
        echo "Unrecognized accelerator line: ${line}" >&2
        exit 1
    fi
    network_ms="${BASH_REMATCH[1]}"
    fps="${BASH_REMATCH[2]}"
    loops="${BASH_REMATCH[3]}"
}

parse_cpu_line() {
    local line="$1"
    if [[ ! "$line" =~ dequantize\ took\ ([0-9]+\.[0-9]+)\ ms,\ ONNX\ Runtime\ took\ ([0-9]+\.[0-9]+)\ ms ]]; then
        echo "Unrecognized CPU line: ${line}" >&2
        exit 1
    fi
    dequantize_ms="${BASH_REMATCH[1]}"
    onnxruntime_ms="${BASH_REMATCH[2]}"
}

append_csv_row() {
    local model="$1"
    local image="$2"
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "$model" "$image" "$network_ms" "$fps" "$loops" "$dequantize_ms" "$onnxruntime_ms" \
        >> "$CSV_PATH"
}

# Read a remote log of "=== image NAME ===" blocks followed by the two timing lines.
ingest_log() {
    local model="$1"
    local log_path="$2"
    local image=""
    local network_line=""
    local cpu_line=""
    local saw=0

    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -z "$line" ] && continue

        if [[ "$line" == "=== image "* ]]; then
            image="${line#=== image }"
            image="${image% ===}"
            network_line=""
            cpu_line=""
            continue
        fi
        if [[ "$line" == network\ took\ * ]]; then
            network_line="$line"
        elif [[ "$line" == dequantize\ took\ * ]]; then
            cpu_line="$line"
        else
            echo "Unexpected line in latency log: ${line}" >&2
            exit 1
        fi

        if [ -n "$network_line" ] && [ -n "$cpu_line" ]; then
            if [ -z "$image" ]; then
                echo "Timing lines arrived before an image marker" >&2
                exit 1
            fi
            parse_network_line "$network_line"
            parse_cpu_line "$cpu_line"
            append_csv_row "$model" "$image"
            saw=$((saw + 1))
            network_line=""
            cpu_line=""
        fi
    done < "$log_path"

    if [ "$saw" -eq 0 ]; then
        echo "No timing lines captured for ${model}" >&2
        exit 1
    fi
    echo "$saw"
}

measure_model() {
    local model="$1"
    local model_vnnx="$2"
    local model_onnx="$3"
    local sample_raw="$4"

    shopt -s nullglob
    local images=( "${REPO_ROOT_DIR}/${sample_raw}"/*.jpg )
    shopt -u nullglob
    if (( ${#images[@]} == 0 )); then
        echo "===> Skipping ${model}: no JPEG images in ${sample_raw}" >&2
        return 0
    fi
    if (( ${#images[@]} > NUM_IMAGES )); then
        images=( "${images[@]:0:NUM_IMAGES}" )
    fi

    local basenames=()
    local img
    for img in "${images[@]}"; do
        basenames+=( "$(basename "$img")" )
    done

    echo "===> ${model}: transferring model and ${#basenames[@]} images..."
    ssh_device "mkdir -p '${DEVICE_SDK_PATH}/${sample_raw}'"
    rsync -az -R \
        -e "ssh -p ${DEVICE_PORT}" \
        "${REPO_ROOT_DIR}/./${model_vnnx}" \
        "${REPO_ROOT_DIR}/./${model_onnx}" \
        "${DEVICE_USERNAME}@${DEVICE_IP}:${DEVICE_SDK_PATH}/"
    rsync -az \
        -e "ssh -p ${DEVICE_PORT}" \
        "${images[@]}" \
        "${DEVICE_USERNAME}@${DEVICE_IP}:${DEVICE_SDK_PATH}/${sample_raw}/"

    echo "===> ${model}: running inference..."
    local log
    log="$(mktemp)"
    ssh_device \
        sh -s -- \
        "$DEVICE_SDK_PATH" \
        "$sample_raw" \
        "$model_vnnx" \
        "$model_onnx" \
        "$APP_NAME" \
        "${basenames[@]}" <<'EOF' > "$log"
set -eu
sdk_path=$1
sample_raw=$2
model_vnnx=$3
model_onnx=$4
app_name=$5
shift 5

cd "$sdk_path"
for base in "$@"; do
    img="${sample_raw}/${base}"
    echo "=== image ${base} ==="
    run_log="/tmp/run-model-latency-$$.log"
    if ! "./apps/${app_name}/run-model" "$model_vnnx" "$img" "$model_onnx" >"$run_log" 2>&1; then
        echo "run-model failed for ${img}" >&2
        cat "$run_log" >&2
        rm -f "$run_log"
        exit 1
    fi
    if ! grep -E 'network took|dequantize took' "$run_log"; then
        echo "Timing lines missing for ${img}" >&2
        tail -n 50 "$run_log" >&2
        rm -f "$run_log"
        exit 1
    fi
    rm -f "$run_log"
done
EOF

    local captured
    captured="$(ingest_log "$model" "$log")"
    rm -f "$log"
    if [ "$captured" -ne "${#basenames[@]}" ]; then
        echo "${model}: captured ${captured} timings, expected ${#basenames[@]}" >&2
        exit 1
    fi
    echo "===> ${model}: recorded ${captured} images"
}

measure_model \
    "mobilepose" \
    "output/embedded-models/ccn1--bijou-rasp-epoch17/ccn1--bijou-rasp-epoch17.vnnx" \
    "output/embedded-models/ccn1--bijou-rasp-epoch17/ccn1--bijou-rasp-epoch17_postprocessing.onnx" \
    "output/images/ccn1-kr-224x224-test-sample/raw"

measure_model \
    "fcos" \
    "output/embedded-models/ccn1--quare-delf-epoch19/ccn1--quare-delf-epoch19.vnnx" \
    "output/embedded-models/ccn1--quare-delf-epoch19/ccn1--quare-delf-epoch19_postprocessing.onnx" \
    "output/images/ccn1-od-384x288-test-sample/raw"

echo "===> Summarizing ${CSV_PATH}..."
VENV_PATH="${REPO_ROOT_DIR}/exporting/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH. Running setup_env.sh to create it..."
    bash "${REPO_ROOT_DIR}/exporting/setup_env.sh"
fi
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
python -m diossa_model_exporter.summarize-latency \
    --csv "$CSV_PATH" \
    --output "$RST_PATH" \
    --device "$DEVICE_IP"

echo "===> CSV: ${CSV_PATH}"
echo "===> RST: ${RST_PATH}"
