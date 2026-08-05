#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
optimizer=$project_root/build/tools/einsumcc-opt/einsumcc-opt

if [ ! -x "$optimizer" ]; then
  echo "missing $optimizer; run scripts/build-mlir.sh first" >&2
  exit 1
fi
PYTHONDONTWRITEBYTECODE=1 python3 "$script_dir/test_cpu_codegen.py"
