# PolarFire SoC: accelerator network plus an ONNX CPU head

This example starts from [`soc-c`](../soc-c) and replaces the built-in C post-processing with an ONNX model executed on the CPU by ONNX Runtime.

The VectorBlox core runs the `.vnnx` network. Each accelerator output is then passed, in order, as an input of `HEAD.onnx`. That split is for graphs that do not fit entirely in the VectorBlox core: the core runs the embedded portion, and the remaining graph runs on the RISC-V application cores.

## Pre-requisites

- The same board setup as [`soc-c`](../soc-c), including the VectorBlox SoC demo design.
- ONNX Runtime built for `riscv64`. This tree already has that build in `ort/output`:
  - headers in `ort/output/include` (`onnxruntime_c_api.h`, `onnxruntime_cxx_api.h`)
  - `ort/output/libonnxruntime.so.1` (SONAME `libonnxruntime.so.1`, ONNX Runtime 1.31.0)
- `libjpeg` and `libatomic` on the target. The ONNX Runtime library is linked against `libatomic.so.1`.

The wheel `ort/output/onnxruntime-1.31.0-cp312-cp312-linux_riscv64.whl` is the same runtime packaged for Python. This example links the shared library directly and does not use the wheel.

## Build

From this directory, on the board or with a RISC-V cross compiler:

```bash
make overlay
make
make stage
```

`make overlay` loads the VectorBlox device tree overlay. If it reports that the overlay already exists, that can be ignored.

`make stage` copies `libonnxruntime.so.1` next to `run-model`. The binary has an `RPATH` of `$ORIGIN`, so the loader finds that copy without `LD_LIBRARY_PATH`. Running from a checkout of this SDK also finds `../../ort/output` through a second `RPATH` entry.

Cross-compile by overriding the compilers. Point `ORT_DIR` at another install if the headers and library are not in `ort/output`:

```bash
make CC=riscv64-unknown-linux-gnu-gcc CXX=riscv64-unknown-linux-gnu-g++ ORT_DIR=/path/to/ort
```

`kit=discovery` and `kit=icicle` select the same interrupt and PDMA options as `soc-c`.

## Run

```bash
./run-model MODEL.vnnx IMAGE.jpg HEAD.onnx
./run-model MODEL.vnnx TEST_DATA HEAD.onnx
```

`TEST_DATA` uses the input and output vectors stored in the VNNX file and checks the accelerator checksum before the ONNX graph runs.

`ORT_NUM_THREADS` sets the ONNX Runtime intra-op thread count. The default is 1.

`WRITE_OUT=1` writes every ONNX output tensor to a JSON file next to the image. `IMAGE.jpg` becomes `IMAGE.json`. Each entry has `name`, `dtype`, `shape`, and `data`.

## What the ONNX graph must look like

ONNX inputs and VNNX outputs are paired by index: VBX output 0 feeds ONNX input 0, and so on. The counts must match.

- A `float32` input is filled with dequantized accelerator values, `(q - zero_point) * scale`, using the scale and zero-point stored in the VNNX file.
- An integer input whose dtype matches the accelerator output (`int8`, `uint8`, `int16`, or `int32`) receives the raw values. Use this when the ONNX graph itself starts with dequantization.
- Static dimensions must equal the VBX shape. A `-1` dimension is filled from the VBX shape.
- If the ranks differ but the element counts match, the same contiguous buffer is viewed with the ONNX shape. Values stay in VBX memory order. Put a `Transpose` in the ONNX graph when the head expects a different layout.

The program prints each VBX output (dtype, shape, scale, zero-point), the ONNX input it was bound to, ONNX Runtime latency, and a short summary of each ONNX output. Float outputs include min, max, argmax, and the top 5 values when the tensor has at most 4096 elements.
