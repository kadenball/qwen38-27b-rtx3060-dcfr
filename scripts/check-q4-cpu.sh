#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$root/third_party/llama.cpp-q4"
lib="$root/build/q4/bin"
"${CXX:-c++}" -std=c++17 -O2 \
    -I"$source_dir/ggml/include" -I"$source_dir/ggml/src" -I"$source_dir/ggml/src/ggml-cpu" \
    "$root/tests/q4-cpu.cpp" -L"$lib" -Wl,-rpath,"$lib" \
    -lggml-cpu -lggml-base -o "$root/build/q4/check-cpu"
for width in 2 3 4 5 6; do
    GGML_CPU_IQ4XS_DOT2=1 GGML_CPU_IQ4XS_MULTI="$width" "$root/build/q4/check-cpu" --graphs
done
