#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
toolchain=$project_root/.deps/llvm-macos-arm64
optimizer=$project_root/build/tools/einsumcc-opt/einsumcc-opt

if [ ! -x "$optimizer" ]; then
  echo "missing $optimizer; run scripts/build-mlir.sh first" >&2
  exit 1
fi

PATH=$toolchain/bin:$project_root/build/tools/einsumcc-opt:$PATH
export PATH

einsumcc-opt "$project_root/test/Dialect/TC/valid.mlir" \
  | FileCheck "$project_root/test/Dialect/TC/valid.mlir" --check-prefix=ROUNDTRIP
einsumcc-opt "$project_root/test/Dialect/TC/valid.mlir" --tc-contract-to-linalg \
  | FileCheck "$project_root/test/Dialect/TC/valid.mlir" --check-prefix=LOWER

diagnostics=$project_root/build/invalid-tc.log
if einsumcc-opt "$project_root/test/Dialect/TC/invalid.mlir" -o /dev/null \
    2>"$diagnostics"; then
  echo "invalid tc.contract unexpectedly passed verification" >&2
  exit 1
fi
FileCheck "$project_root/test/Dialect/TC/invalid.mlir" <"$diagnostics"

for source in invalid_extent invalid_broadcast invalid_zero_extent; do
  diagnostics=$project_root/build/$source.log
  if einsumcc-opt "$project_root/test/Dialect/TC/$source.mlir" -o /dev/null \
      2>"$diagnostics"; then
    echo "$source unexpectedly passed verification" >&2
    exit 1
  fi
  FileCheck "$project_root/test/Dialect/TC/$source.mlir" <"$diagnostics"
done
echo "MLIR dialect and lowering tests passed"
