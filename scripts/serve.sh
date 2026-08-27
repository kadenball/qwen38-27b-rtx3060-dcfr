#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_server="${LLAMA_SERVER:-$repo_root/third_party/llama.cpp/build-cuda/bin/llama-server}"
model_path="${MODEL_PATH:-}"
host="${HOST:-127.0.0.1}"
port="${PORT:-8080}"

if [[ -z "$model_path" ]]; then
    echo "Set MODEL_PATH to Qwen3.8-27B-UD-IQ3_XXS.gguf." >&2
    exit 2
fi

for required_file in "$llama_server" "$model_path"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file is missing: $required_file" >&2
        exit 2
    fi
done

if [[ -n "${CUDA_TOOLKIT_ROOT:-}" ]]; then
    export LD_LIBRARY_PATH="$CUDA_TOOLKIT_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export LLAMA_GDN_TRANSACTIONAL_REPLAY=1
export GGML_OP_OFFLOAD_MIN_BATCH=2

exec "$llama_server" \
    -m "$model_path" --alias qwen3.8-27b-iq3-64k-dcfr \
    -c 65536 --parallel 1 \
    -dev CUDA0 --fit off --n-gpu-layers 50 \
    --override-tensor 'blk\.(10|11|12|13|14|15|16)\..*=CUDA0' \
    --load-mode none \
    -ctk q4_0 -ctv q4_0 \
    -b 16 -ub 16 \
    -t 6 -tb 6 \
    --spec-type draft-mtp \
    --spec-draft-n-max 6 \
    --spec-draft-p-min 0 \
    --spec-draft-type-k q4_0 \
    --spec-draft-type-v q4_0 \
    --spec-draft-threads 6 \
    --spec-draft-threads-batch 6 \
    -fa on --no-mmproj --reasoning off --jinja \
    --cache-ram 0 --no-cache-idle-slots --no-ui \
    --host "$host" --port "$port" --metrics \
    "$@"
