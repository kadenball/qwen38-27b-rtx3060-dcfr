#!/usr/bin/env bash
# Separate source/build trees leave historical releases and running servers intact.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
variant="${Q3_VARIANT:-candidate}"
case "$variant" in
    candidate) pins=Q3_SOURCE_SHA256SUMS ;;
    baseline) pins=Q3_BASELINE_SOURCE_SHA256SUMS ;;
    *) echo 'Q3_VARIANT must be candidate or baseline.' >&2; exit 2 ;;
esac
source_dir="$root/third_party/llama.cpp-q3-$variant"
build_dir="$root/build/q3-$variant"
revision=c060ca974c773c7c3d17fd1b66dc9d312bc292c0
cuda_root="${CUDA_TOOLKIT_ROOT:-/usr/local/cuda}"
for tool in curl tar patch cmake ninja sha256sum "${CC:-cc}" "${CXX:-c++}"; do
    command -v "$tool" >/dev/null || { echo "Missing prerequisite: $tool" >&2; exit 2; }
done
[[ "${BUILD_JOBS:-1}" =~ ^[1-9][0-9]*$ ]] || { echo 'BUILD_JOBS must be positive.' >&2; exit 2; }
cd "$root"
sha256sum --check --status SHA256SUMS
if [[ ! -d "$source_dir" ]]; then
    mkdir -p "$root/third_party"
    staging="$(mktemp -d "$root/third_party/q3-$variant-source.XXXXXX")"
    curl -fL --retry 3 "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$revision" -o "$staging/source.tar.gz"
    echo "34a2d876cf9b12867742193cc78a24b23eb4cc6f669816f6f529383909ffd7ad  $staging/source.tar.gz" | sha256sum -c -
    candidate="$staging/llama.cpp-$revision"
    tar -xzf "$staging/source.tar.gz" -C "$staging"
    overlays=(llama-cpp-dcfr-research.patch llama-cpp-q4-interactive.patch llama-cpp-q3-cpu.patch llama-cpp-q3-prefetch.patch)
    [[ "$variant" != candidate ]] || overlays+=(llama-cpp-q3-cache-safe.patch)
    for overlay in "${overlays[@]}"; do
        patch --batch --fuzz=0 -p1 -d "$candidate" -i "$root/patches/$overlay"
    done
    (cd "$candidate" && sha256sum -c "$root/patches/$pins")
    mv "$candidate" "$source_dir"
    echo "Source archive retained in $staging"
else
    (cd "$source_dir" && sha256sum -c "$root/patches/$pins")
fi
[[ "${PREPARE_ONLY:-0}" != 1 ]] || exit 0
[[ -x "$cuda_root/bin/nvcc" ]] || { echo 'Set CUDA_TOOLKIT_ROOT to a CUDA toolkit containing bin/nvcc.' >&2; exit 2; }
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC:-cc}" -DCMAKE_CXX_COMPILER="${CXX:-c++}" \
    -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="${CXX:-c++}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-native}" -DCUDAToolkit_ROOT="$cuda_root" \
    -DCMAKE_BUILD_RPATH="$cuda_root/lib64" -DCMAKE_DISABLE_FIND_PACKAGE_Git=TRUE \
    -DLLAMA_BUILD_COMMIT="c060ca974c77-routeweaver-q3-$variant" -DLLAMA_BUILD_NUMBER=1 \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF \
    -DGGML_CUDA=ON -DGGML_NATIVE=ON
cmake --build "$build_dir" --target llama-server --parallel "${BUILD_JOBS:-1}"
"$build_dir/bin/llama-server" --version
