# diossa-vbx

Host-side export and on-device run of the DIOSSA CCN1 vision chain on a Microchip PolarFire SoC with the VectorBlox accelerator.

The payload estimates the 6-DoF pose of a target spacecraft from a monocular image. The chain is two networks in series: FCOS for object detection, then MobilePose for keypoint regression. In CCN1 those networks were trained in floating point, refined with quantization-aware training, and compiled onto a Xilinx DPU (Vitis-AI 3.5, UltraScale+ MPSoC). This repository keeps the CCN1 architectures and weights, and compiles the same quantization-aware checkpoints for VectorBlox SDK 3.1.1 (V1000 core, no weight compression).

The trade-off against the Xilinx baseline is written up in [`reports/PolarFire_Xilinx_Tradeoff/PolarFire_Xilinx_Tradeoff.rst`](reports/PolarFire_Xilinx_Tradeoff/PolarFire_Xilinx_Tradeoff.rst) (PDF next to the source).

## Models

Each network is split the same way it was on Xilinx. The convolutional edge runs on the accelerator. The tail that VectorBlox cannot compile (decoding, NMS, heatmap-to-coordinate) stays on the RISC-V application cores as ONNX, executed by ONNX Runtime. Accelerator outputs are dequantized on the CPU and then fed to that graph.

| Role | Checkpoint | Input (W × H) | Accelerator binary | CPU tail |
| --- | --- | --- | --- | --- |
| Keypoint regression | `ccn1--bijou-rasp-epoch17` (QAT, 20 keypoints) | 224 × 224 | `.vnnx` | coords, normalized heatmaps |
| Object detection | `ccn1--quare-delf-epoch19` (QAT) | 384 × 288 | `.vnnx` | boxes, scores, labels |

The float parents (`ccn1--blest-harl` for MobilePose, `ccn1--licit-weal` for FCOS) are kept under `data/models/` as the accuracy reference. Post-training quantization of those float weights did not produce a finite INT8 scale, so the binaries on the board are the QAT checkpoints. FCOS is exported through `FCOS_PTQ`, which uses `BatchNorm2d` in place of `GroupNorm`, because the quantizer does not accept group norm.

Checkpoints live in `data/models/`. The two FCOS `.ckpt` files are Git LFS objects. After clone:

```bash
git lfs install
git lfs pull
```

## Layout

```
apps/single-model-runner/   On-target C++ runner: .vnnx on VectorBlox, HEAD.onnx on ONNX Runtime
data/                       CCN1 checkpoints, a few earlier .vnnx/.onnx files, camera and 3D model
exporting/                  diossa_model_exporter: ONNX export, calibration, sample pull, comparison
output/                     Generated artifacts (most of this is gitignored)
reports/                    PolarFire/Xilinx note, CCN1 DJF, Xilinx latency logs
scripts/                    Numbered host pipeline, 00 through 09
third-party/nn-models/      PyTorch FCOS and MobilePose used at export time
third-party/onnxruntime/    Docker cross-compile of ONNX Runtime for riscv64
third-party/vbx-sdk/        VectorBlox SDK 3.1.1 (onnx2tf path, tflite_preprocess, vnnx_compile)
```

`output/` holds what the scripts produce:

| Path | Contents | In git |
| --- | --- | --- |
| `output/images/ccn1-kr-224x224-test-sample/` | MobilePose sample: `raw/`, `gt/`, `annot/`, `pred/`, `pred_annot/` | yes |
| `output/images/ccn1-od-384x288-test-sample/` | FCOS sample, same subdirectories | yes |
| `output/embedded-models/<ckpt>/` | ONNX edge, ONNX tail, INT8 TFLite, `.vnnx`, calibration `.npy` | no |
| `output/latency/` | `latency.csv` and an RST summary | no |
| `output/sysroot/` | Headers and libs rsynced from the board | no |
| `output/onnxruntime-riscv64/` | Cross-compiled ONNX Runtime | no |

A sample directory looks like this:

```
raw/         JPEG at the network input size
gt/          JSON keypoints and boxes
annot/       raw image with ground truth drawn
pred/        JSON tensors written by run-model (WRITE_OUT=1)
pred_annot/  ground truth in green, prediction in red
```

`data/camera.json` and `data/model-3d-points.json` are the camera intrinsics and the spacecraft keypoint model used by `exporting/src/diossa_model_exporter/pnp_solver/pose_estimator.py`. The numbered scripts stop at detection and keypoints. Image decode and the pose solver are outside the latency numbers in the note.

## Prerequisites

- Ubuntu host with Docker, `rsync`, and SSH to the PolarFire board.
- [`uv`](https://docs.astral.sh/uv/) for the export environment (Python 3.12).
- Git LFS.
- The VectorBlox SoC demo design already running on the board. `scripts/00-set-up-env.sh` calls `third-party/vbx-sdk/install_dependencies.sh`, which needs sudo.
- Access to the internal Dataset Viewer (`http://datasets.int.lmo.space`) only if you regenerate the 100-image samples. The checked-in `raw/`, `gt/`, and `annot/` trees are there so the rest of the pipeline runs without that service.
- Access to `registry.gitlab.com/lmo-space/ios/development/diossa-ccn/docs:latest` only for the PDF build.

The board defaults are in every device script and can be overridden:

| Variable | Default |
| --- | --- |
| `DEVICE_IP` | `192.168.20.6` |
| `DEVICE_PORT` | `22` |
| `DEVICE_USERNAME` | `root` |
| `DEVICE_SDK_PATH` | `/root/vbx-sdk` |

## Pipeline

Run from the repository root, in order, the first time you bring up a board. Later steps can be re-run on their own once their inputs exist.

| Script | What it does |
| --- | --- |
| `scripts/00-set-up-env.sh` | Install VectorBlox SDK dependencies, create `third-party/vbx-sdk/vbx_env`, and create `exporting/.venv` |
| `scripts/01-pull-device-sysroot.sh` | Rsync `/lib`, `/usr/include`, and `/usr/lib` from the board into `output/sysroot` |
| `scripts/02-transfer-sdk-to-device.sh` | Copy SDK `drivers/`, `apps/`, and `lib/` to the board |
| `scripts/03-compile-onnxruntime.sh` | Cross-compile ONNX Runtime in Docker and rsync headers and `libonnxruntime` to the board |
| `scripts/04-embed-models.sh` | Pull the CCN1 samples (if needed) and export both QAT networks to `.vnnx` plus a CPU ONNX tail |
| `scripts/05-transfer-compile-and-test-app.sh` | Build `single-model-runner` on the board and smoke-test MobilePose on one image |
| `scripts/06-run-inference-on-sample.sh` | Run one model over its 100-image sample and pull `pred/*.json` back |
| `scripts/07-run-comparisons.sh` | Score keypoints (L2, with optional GT–prediction lines) and boxes (IoU of the top detection), and write `pred_annot/` |
| `scripts/08-measure-latency.sh` | Time accelerator, dequantize, and ONNX Runtime. Default is 50 images (`NUM_IMAGES`) |
| `scripts/09-build-doc.sh` | Build the trade-off PDF with the internal Sphinx image |

`06` is wired to one model at a time. `05` smoke-tests MobilePose. `07` compares whichever `pred/` directory already has JSON files and skips the other. `08` measures both models.

`SKIP_EXISTING=1` on `06` leaves images whose JSON is already on the board untouched.

## Embedding

`exporting/scripts/export-mobilepose.sh` and `exporting/scripts/export-fcos.sh` set the checkpoint, input shape, class, remapper, and ImageNet mean and standard deviation, then call `exporting/scripts/export-pytorch-model.sh`. That script is idempotent: each artifact is skipped when the file is already present.

1. Load the checkpoint. A remapper drops Vitis-AI fake-quantizer keys and renames batch-norm weights so they match the export graph. Two ONNX files are written: the accelerator edge and the CPU tail.
2. Build a calibration array of 100 images from the matching sample, resized to the network input and scaled to 0–1.
3. Simplify the edge ONNX with `onnxsim`.
4. Convert and quantize to a fully integer INT8 TFLite graph with `onnx2tf`. The calibration array sets the per-tensor scale and zero-point. Mean and standard deviation are applied there as well.
5. Run `tflite_preprocess` so the accelerator consumes pixels in 0–255 with the same normalization the network was trained with, and insert the uint8-to-int8 quantize at the input.
6. Compile with `vnnx_compile -s V1000 -c ncomp`.

Outputs land in `output/embedded-models/<checkpoint-name>/`.

`single-model-runner` pairs VNNX outputs with ONNX inputs by index when the shapes already agree. When they do not, each ONNX input takes the earliest unused VBX output of the same shape, in order. FCOS needs that reorder: the core emits the fifteen maps in execution order, and the CPU graph expects every box map, then every class map, then every centerness map. See [`apps/single-model-runner/README.md`](apps/single-model-runner/README.md).

On the board:

```bash
WRITE_OUT=1 ./apps/single-model-runner/run-model MODEL.vnnx IMAGE.jpg HEAD.onnx
```

`WRITE_OUT=1` writes one JSON file per image, next to the JPEG, with `name`, `dtype`, `shape`, and `data` for every ONNX output. `ORT_NUM_THREADS` sets the ONNX Runtime intra-op thread count (default 1).

## Comparison

MobilePose coordinates come back on the 56 × 56 heatmap and are scaled onto the 224 × 224 input before the Euclidean error is computed. FCOS boxes are already in input-image pixels. Each frame has one object, so the box with the highest score is the detection that is scored.

```bash
./scripts/07-run-comparisons.sh
```

That calls `exporting/scripts/compare-mobilepose-sample.sh --pair-lines` and `exporting/scripts/compare-fcos-sample.sh`. Yellow lines in the keypoint overlays connect each ground-truth point to its prediction.

## Report

The note records latency on both platforms and qualitative overlays from the PolarFire run. Accuracy on the full CCN1 test set (IoU, keypoint error) is an open item: the overlays show MobilePose keypoints swapped on some frames, and FCOS boxes that fall inside the spacecraft but do not cover the ground-truth extent.

On the 50-image PolarFire measurement (V1000, no compression), mean end-to-end time is 38.33 ms for MobilePose and 409.85 ms for FCOS. Added together that is 448 ms, which meets the 1 Hz application budget. The Xilinx sums from the earlier campaign are 59.8 ms and 56.3 ms. Image decode and the pose solver are excluded on both sides. The full tables, the Xilinx FLOAT-versus-QAT accuracy baseline, and the open items are in the RST.

```bash
./scripts/09-build-doc.sh
```

The PDF is copied to `reports/PolarFire_Xilinx_Tradeoff/PolarFire_Xilinx_Tradeoff.pdf`. `reports/diossa-ccn1-djf/` is the CCN1 design justification file the note cites. `reports/xilinx-latency-benchmarks/` holds the DPU and ONNX Runtime logs those Xilinx times came from.
