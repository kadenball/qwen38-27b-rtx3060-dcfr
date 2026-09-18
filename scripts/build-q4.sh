#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$root/third_party/llama.cpp-q4"
build_dir="$root/build/q4"
revision=c060ca974c773c7c3d17fd1b66dc9d312bc292c0
cuda_root="${CUDA_TOOLKIT_ROOT:-/usr/local/cuda}"
for tool in curl tar patch cmake ninja sha256sum "${CC:-cc}" "${CXX:-c++}"; do
    command -v "$tool" >/dev/null || { echo "Missing prerequisite: $tool" >&2; exit 2; }
done
[[ -x "$cuda_root/bin/nvcc" ]] || { echo 'Set CUDA_TOOLKIT_ROOT to a CUDA toolkit containing bin/nvcc.' >&2; exit 2; }
cd "$root"
sha256sum -c SHA256SUMS
if [[ ! -d "$source_dir" ]]; then
    mkdir -p "$root/third_party"
    staging="$(mktemp -d "$root/third_party/q4-source.XXXXXX")"
    curl -fL --retry 3 "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$revision" -o "$staging/source.tar.gz"
    echo "34a2d876cf9b12867742193cc78a24b23eb4cc6f669816f6f529383909ffd7ad  $staging/source.tar.gz" | sha256sum -c -
    tar -xzf "$staging/source.tar.gz" -C "$staging"
    candidate="$staging/llama.cpp-$revision"
    patch --batch --fuzz=0 -p1 -d "$candidate" -i "$root/patches/llama-cpp-dcfr-research.patch"
    patch --batch --fuzz=0 -p1 -d "$candidate" -i "$root/patches/llama-cpp-q4-interactive.patch"
    (cd "$candidate" && sha256sum -c "$root/patches/Q4_SOURCE_SHA256SUMS")
    mv "$candidate" "$source_dir"
    echo "Download archive retained in $staging"
else
    (cd "$source_dir" && sha256sum -c "$root/patches/Q4_SOURCE_SHA256SUMS")
fi
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC:-cc}" -DCMAKE_CXX_COMPILER="${CXX:-c++}" \
    -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="${CXX:-c++}" \
    -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDAToolkit_ROOT="$cuda_root" \
    -DCMAKE_BUILD_RPATH="$cuda_root/lib64" \
    -DCMAKE_DISABLE_FIND_PACKAGE_Git=TRUE \
    -DLLAMA_BUILD_COMMIT=c060ca974c77-q4-interactive -DLLAMA_BUILD_NUMBER=1 \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF \
    -DGGML_CUDA=ON -DGGML_NATIVE=ON
cmake --build "$build_dir" --target llama-server --parallel "${BUILD_JOBS:-2}"
"$build_dir/bin/llama-server" --version
