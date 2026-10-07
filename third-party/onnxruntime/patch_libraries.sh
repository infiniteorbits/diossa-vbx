#!/bin/bash
set -e

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT_DIR}/output/onnxruntime-riscv64"

if [ ! -d "$OUTPUT_DIR/raw" ] || ! ls "$OUTPUT_DIR/raw/"libonnxruntime.so* >/dev/null 2>&1; then
  echo "Error: No raw libonnxruntime.so* found in $OUTPUT_DIR/raw/. Run ./run_build.sh first."
  exit 1
fi

docker run --rm -it \
  -v "$OUTPUT_DIR:/output" \
  ort-riscv-builder bash -c '
    set -e

    echo "===> Post-processing ONNX Runtime C/C++ SDK bundle..."

    # 1. Clean previous outputs and create clean structure
    rm -rf /output/include /output/lib /output/*.so*
    mkdir -p /output/include /output/lib

    # 2. Locate RISC-V target libraries from cross-compiler path
    REAL_LIBSTDCXX=$(find /usr/riscv64-linux-gnu /usr/lib/gcc-cross/riscv64-linux-gnu -name "libstdc++.so.6*" -type f ! -name "*.py" 2>/dev/null | head -n 1)
    REAL_LIBATOMIC=$(find /usr/riscv64-linux-gnu /usr/lib/gcc-cross/riscv64-linux-gnu -name "libatomic.so.1*" -type f ! -name "*.py" 2>/dev/null | head -n 1)

    # 3. Copy shared libraries strictly into /output/lib/
    cp -d /output/raw/libonnxruntime.so* /output/lib/
    cp -L "$REAL_LIBSTDCXX" /output/lib/libstdc++.so.6
    cp -L "$REAL_LIBATOMIC" /output/lib/libatomic.so.1

    # 4. Create unversioned symlinks so standard -lstdc++ and -latomic match
    ln -sf libstdc++.so.6 /output/lib/libstdc++.so
    ln -sf libatomic.so.1 /output/lib/libatomic.so

    chmod 755 /output/lib/libstdc++.so.6 /output/lib/libatomic.so.1

    # 5. Patch RPATH to $ORIGIN on all shared libraries in /output/lib/
    echo "===> Patching RPATH to '\''\$ORIGIN'\''..."
    for so_file in /output/lib/*.so*; do
      if [ -f "$so_file" ] && [ ! -L "$so_file" ]; then
        if head -c 4 "$so_file" | grep -q "ELF"; then
          patchelf --set-rpath '\''$ORIGIN'\'' "$so_file"
        fi
      fi
    done

    # 6. Export C/C++ API headers
    echo "===> Exporting C/C++ API headers to /output/include/..."
    cp -r /onnxruntime/include/onnxruntime/core/session/* /output/include/

    echo "===> Success! SDK layout created at $OUTPUT_DIR:"
    echo "  - Libraries: $OUTPUT_DIR/lib/"
    echo "  - Headers  : $OUTPUT_DIR/include/"
'
