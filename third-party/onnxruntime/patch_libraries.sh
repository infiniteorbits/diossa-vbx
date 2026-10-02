#!/bin/bash
set -e

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

OUTPUT_DIR="${REPO_ROOT_DIR}/output/onnxruntime-riscv64"

docker run --rm -it \
  -v "$OUTPUT_DIR:/output" \
  ort-riscv-builder bash -c '
    set -e

    # 1. Create target SDK directories
    mkdir -p /output/lib /output/include

    # 2. Locate built libonnxruntime.so files
    SRC_SO=$(find /output/raw /output -maxdepth 2 -name "libonnxruntime.so*" -type f 2>/dev/null | head -n 1)
    if [ -z "$SRC_SO" ]; then
      echo "Error: No libonnxruntime.so* found in /output or /output/raw!"
      exit 1
    fi

    echo "===> Copying ONNX Runtime shared libraries..."
    cp -d $(dirname "$SRC_SO")/libonnxruntime.so* /output/lib/ 2>/dev/null || cp "$SRC_SO" /output/lib/

    # 3. Locate RISC-V target libraries from container cross-compiler path
    REAL_LIBSTDCXX=$(find /usr/riscv64-linux-gnu /usr/lib/gcc-cross/riscv64-linux-gnu -name "libstdc++.so.6*" -type f ! -name "*.py" 2>/dev/null | head -n 1)
    REAL_LIBATOMIC=$(find /usr/riscv64-linux-gnu /usr/lib/gcc-cross/riscv64-linux-gnu -name "libatomic.so.1*" -type f ! -name "*.py" 2>/dev/null | head -n 1)

    echo "===> Found RISC-V libstdc++: $REAL_LIBSTDCXX"
    echo "===> Found RISC-V libatomic: $REAL_LIBATOMIC"

    # 4. Copy runtime dependencies into /output/lib/
    cp -L "$REAL_LIBSTDCXX" /output/lib/libstdc++.so.6
    cp -L "$REAL_LIBATOMIC" /output/lib/libatomic.so.1
    chmod 755 /output/lib/libstdc++.so.6 /output/lib/libatomic.so.1

    # 5. Patch RPATH to $ORIGIN for all shared objects in /output/lib/
    echo "===> Patching RPATH to '\$ORIGIN'..."
    for so_file in /output/lib/*.so*; do
      if [ -f "$so_file" ] && [ ! -L "$so_file" ]; then
        if file "$so_file" | grep -q "ELF"; then
          patchelf --set-rpath '\''$ORIGIN'\'' "$so_file"
          echo "  Updated: $(basename "$so_file") -> RPATH=$(patchelf --print-rpath "$so_file")"
        fi
      fi
    done

    # 6. Extract C/C++ API headers
    echo "===> Exporting C/C++ API headers to /output/include/..."
    cp -r /onnxruntime/include/onnxruntime/core/session/* /output/include/

    echo "===> Verification of /output/lib/ contents:"
    ls -la /output/lib/

    echo "===> Success! C/C++ SDK bundle created at: $OUTPUT_DIR"
    echo "  - Libraries: $OUTPUT_DIR/lib/"
    echo "  - Headers  : $OUTPUT_DIR/include/"
'