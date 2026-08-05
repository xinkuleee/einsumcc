#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
toolchain=$project_root/.deps/llvm-macos-arm64
cmake_bin=${CMAKE:-cmake}
ninja_bin=${NINJA:-ninja}

if [ -x "$project_root/.deps/build-tools/bin/cmake" ]; then
  cmake_bin=$project_root/.deps/build-tools/bin/cmake
  export PYTHONPATH=$project_root/.deps/build-tools${PYTHONPATH:+:$PYTHONPATH}
elif [ -x "$project_root/.deps/build-venv/bin/cmake" ]; then
  cmake_bin=$project_root/.deps/build-venv/bin/cmake
fi
if [ -x "$project_root/.deps/build-tools/bin/ninja" ]; then
  ninja_bin=$project_root/.deps/build-tools/bin/ninja
elif [ -x "$project_root/.deps/build-venv/bin/ninja" ]; then
  ninja_bin=$project_root/.deps/build-venv/bin/ninja
fi

"$cmake_bin" -S "$project_root" -B "$project_root/build" -G Ninja \
  -DMLIR_DIR="$toolchain/lib/cmake/mlir" \
  -DLLVM_DIR="$toolchain/lib/cmake/llvm" \
  -DCMAKE_MAKE_PROGRAM="$ninja_bin" \
  -DCMAKE_C_COMPILER="$toolchain/bin/clang" \
  -DCMAKE_CXX_COMPILER="$toolchain/bin/clang++" \
  -DCMAKE_BUILD_TYPE=Release
"$cmake_bin" --build "$project_root/build" --target einsumcc-opt
