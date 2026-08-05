#!/bin/sh
set -eu

# Keep this revision aligned with the local Triton checkout used for the A100
# continuation. The archive is an official Triton LLVM/MLIR build for arm64
# macOS; source and checksum are intentionally reviewable here.
archive_name=llvm-850a2b1b-macos-arm64-1.tar.gz
archive_url=https://oaitriton.blob.core.windows.net/public/llvm-builds/$archive_name
archive_sha256=2a7b0bc85b022009a5753aae3c4a810f9eafd4224697d6fd4cdeb5041b3a6bd8

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
dependency_root=$project_root/.deps
install_root=$dependency_root/llvm-macos-arm64
archive_path=$dependency_root/$archive_name

if [ -x "$install_root/bin/mlir-translate" ] && [ -f "$install_root/lib/cmake/mlir/MLIRConfig.cmake" ]; then
  echo "MLIR toolchain already available at $install_root"
  exit 0
fi

mkdir -p "$dependency_root"
if [ ! -f "$archive_path" ]; then
  curl --fail --location --output "$archive_path" "$archive_url"
fi

actual_sha256=$(shasum -a 256 "$archive_path" | awk '{print $1}')
if [ "$actual_sha256" != "$archive_sha256" ]; then
  echo "checksum mismatch for $archive_path" >&2
  echo "expected: $archive_sha256" >&2
  echo "actual:   $actual_sha256" >&2
  exit 1
fi

staging_root=$dependency_root/llvm-staging
if [ -e "$staging_root" ]; then
  echo "stale staging directory exists: $staging_root" >&2
  echo "remove it manually after checking its contents" >&2
  exit 1
fi
mkdir -p "$staging_root"
tar -xzf "$archive_path" -C "$staging_root"

# Triton archives currently contain their payload at the archive root. Keep a
# single stable install path even if a future archive adds one wrapper folder.
if [ -x "$staging_root/bin/mlir-translate" ] && [ -f "$staging_root/lib/cmake/mlir/MLIRConfig.cmake" ]; then
  mv "$staging_root" "$install_root"
else
  candidate=$(find "$staging_root" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  if [ -z "$candidate" ] || [ ! -x "$candidate/bin/mlir-translate" ] || [ ! -f "$candidate/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo "archive does not contain a recognized MLIR development toolchain" >&2
    exit 1
  fi
  mv "$candidate" "$install_root"
  rmdir "$staging_root"
fi

echo "Installed MLIR toolchain at $install_root"
