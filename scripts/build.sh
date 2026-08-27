#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${LLAMA_SOURCE_DIR:-$repo_root/third_party/llama.cpp}"
build_dir="${LLAMA_BUILD_DIR:-$source_dir/build-cuda}"
base_revision="$(tr -d '[:space:]' < "$repo_root/patches/LLAMA_CPP_BASE_COMMIT")"
research_patch="$repo_root/patches/llama-cpp-dcfr-research.patch"
cuda_root="${CUDA_TOOLKIT_ROOT:-/usr/local/cuda}"
cuda_arch="${CUDA_ARCHITECTURES:-86}"
build_jobs="${BUILD_JOBS:-6}"

if [[ ! -f "$research_patch" ]]; then
    echo "Required research patch is missing: $research_patch" >&2
    exit 2
fi

if [[ ! -f "$source_dir/CMakeLists.txt" ]]; then
    if [[ -d "$source_dir" ]] && find "$source_dir" -mindepth 1 -print -quit | grep -q .; then
        echo "Refusing to overwrite non-empty source directory: $source_dir" >&2
        exit 2
    fi

    archive_path="$(mktemp /tmp/qwen38-dcfr-llama.XXXXXX.tar.gz)"
    trap 'rm -f "$archive_path"' EXIT
    mkdir -p "$source_dir"
    curl -fL --retry 5 --retry-all-errors --connect-timeout 30 \
        "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$base_revision" \
        -o "$archive_path"
    tar -xzf "$archive_path" --strip-components=1 -C "$source_dir"
    rm -f "$archive_path"
    trap - EXIT

    patch --batch --forward -p1 -d "$source_dir" --input="$research_patch"
fi

for command_name in cmake "${CC:-cc}" "${CXX:-c++}"; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command is missing: $command_name" >&2
        exit 2
    fi
done

if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
    echo "CUDA nvcc was not found at $cuda_root/bin/nvcc" >&2
    echo "Set CUDA_TOOLKIT_ROOT to your CUDA installation." >&2
    exit 2
fi

generator_args=()
if command -v ninja >/dev/null 2>&1; then
    generator_args=(-G Ninja)
fi

cmake \
    -S "$source_dir" \
    -B "$build_dir" \
    "${generator_args[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC:-cc}" \
    -DCMAKE_CXX_COMPILER="${CXX:-c++}" \
    -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" \
    -DCUDAToolkit_ROOT="$cuda_root" \
    -DLLAMA_BUILD_COMMIT="${base_revision:0:12}-dcfr" \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_CURL=OFF \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON

cmake --build "$build_dir" --target llama-server llama-cli llama-bench -j "$build_jobs"
"$build_dir/bin/llama-server" --version
