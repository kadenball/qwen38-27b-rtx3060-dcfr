#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/q3-env.sh"
source_dir="$root/third_party/llama.cpp-q3-$variant"
lib="$build_dir/bin"
[[ -f "$lib/libllama.so" ]] || { echo 'Build Q3 first.' >&2; exit 2; }
includes=(-I"$source_dir/include" -I"$source_dir/src" -I"$source_dir/common"
    -I"$source_dir/ggml/include" -I"$source_dir/ggml/src" -I"$source_dir/ggml/src/ggml-cpu")
link=(-L"$lib" -Wl,-rpath,"$lib")
"${CXX:-c++}" -std=c++17 -O2 "${includes[@]}" "$root/tests/q3-cpu.cpp" "${link[@]}" -lggml-cpu -lggml-base -o "$build_dir/check-cpu"
for width in 2 3 4 5 6; do
    printf 'Checking IQ3 multi-query width %s\n' "$width"
    GGML_CPU_IQ3XXS_MULTI="$width" "$build_dir/check-cpu" --graphs
done
"${CXX:-c++}" -std=c++17 -O2 "${includes[@]}" "$root/tests/q3-prefetch-cleanup.cpp" "${link[@]}" -lggml-base -o "$build_dir/check-prefetch"
"$build_dir/check-prefetch"
if [[ "$variant" != baseline ]]; then
    "${CXX:-c++}" -std=c++17 -O2 "${includes[@]}" "$root/tests/q3-state.cpp" "${link[@]}" -lllama -lggml -lggml-base -o "$build_dir/check-state"
    CUDA_VISIBLE_DEVICES=-1 "$build_dir/check-state"
fi
if [[ "$variant" == fast ]]; then
    "${CXX:-c++}" -std=c++17 -O2 "${includes[@]}" "$root/tests/q3-rejection.cpp" "${link[@]}" -lllama-common -lllama -lggml -lggml-base -o "$build_dir/check-rejection"
    CUDA_VISIBLE_DEVICES=-1 "$build_dir/check-rejection"
fi
"${CXX:-c++}" -std=c++17 -O2 "${includes[@]}" "$root/tests/q3-model.cpp" "${link[@]}" -lllama-common -lllama -lggml -lggml-base -o "$build_dir/check-model"
if [[ "${1:-}" == --model ]]; then
    [[ "$variant" != baseline ]] || { echo 'Exact checkpoint test requires candidate or fast.' >&2; exit 2; }
    args=(-m "$model" -c 256 -b 128 -ub 128 -t "$threads" -tb "$threads" -ngl all --fit off --load-mode none -ctk q4_0 -ctv q4_0 -fa on)
    ((host_blocks == 0)) || args+=(--override-tensor "$tensor_pattern")
    "$build_dir/check-model" "${args[@]}"
else
    [[ $# == 0 ]] || { echo 'Usage: check-q3.sh [--model]' >&2; exit 2; }
    echo 'Model-free checks passed. Run with --model only when the GPU is idle.'
fi
