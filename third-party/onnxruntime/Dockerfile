FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:/opt/protoc/bin:/root/.local/bin:${PATH}"

# Install cross-compiler, QEMU, unzip, flatc, patchelf, and zip
RUN apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    git \
    curl \
    ca-certificates \
    unzip \
    patchelf \
    zip \
    gcc-riscv64-linux-gnu \
    g++-riscv64-linux-gnu \
    qemu-user-static \
    flatbuffers-compiler \
 && rm -rf /var/lib/apt/lists/*

# Symlink GCC binaries to match ONNX Runtime expected toolchain names
RUN ln -s /usr/bin/riscv64-linux-gnu-gcc /usr/bin/riscv64-unknown-linux-gnu-gcc \
 && ln -s /usr/bin/riscv64-linux-gnu-g++ /usr/bin/riscv64-unknown-linux-gnu-g++

# Install uv, create virtual environment at /opt/venv, and install build dependencies
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && uv python install 3.12 \
 && uv venv /opt/venv --python 3.12 \
 && uv pip install numpy setuptools wheel packaging flatbuffers cmake

# Clone ONNX Runtime inside the image
WORKDIR /workspace
RUN git clone --recursive https://github.com/microsoft/onnxruntime.git /onnxruntime \
 && cd /onnxruntime \
 && git checkout v1.30.0 \
 && git submodule update --init --recursive

WORKDIR /onnxruntime
