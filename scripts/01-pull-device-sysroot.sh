#!/bin/bash
set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEVICE_IP="${DEVICE_IP:-192.168.20.6}"
DEVICE_PORT="${DEVICE_PORT:-22}"
DEVICE_USERNAME="${DEVICE_USERNAME:-root}"
SYSROOT="${REPO_ROOT_DIR}/output/sysroot"

mkdir -p "$SYSROOT"

# Mirror only the link/compile inputs. Symlinks must stay symlinks
# (libc.so -> libc.so.6). Do not pass -L / --copy-links.
rsync -aH --numeric-ids --info=progress2 \
  --include='/lib/***' \
  --include='/lib64/***' \
  --include='/usr/' \
  --include='/usr/include/***' \
  --include='/usr/lib/***' \
  --include='/usr/lib64/***' \
  --exclude='*' \
  -e "ssh -p $DEVICE_PORT" \
  "${DEVICE_USERNAME}@${DEVICE_IP}:/" \
  "${SYSROOT}/"