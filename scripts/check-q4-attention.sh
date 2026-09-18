#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$root/third_party/llama.cpp-q4"
lib="$root/build/q4/bin"
"${CXX:-c++}" -std=c++17 -O2 \
    -I"$source_dir/ggml/include" \
    "$root/tests/q4-attention-allocation.cpp" -L"$lib" -Wl,-rpath,"$lib" \
    -lggml-cuda -lggml-base -o "$root/build/q4/check-attention-allocation"
GGML_CUDA_Q4_FA_HEADSPLIT=1 GGML_CUDA_Q4_FA_VEC=1 \
    "$root/build/q4/check-attention-allocation" headsplit
