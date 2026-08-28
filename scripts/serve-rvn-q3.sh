#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_server="${LLAMA_SERVER:-$repo_root/third_party/llama.cpp/build-cuda/bin/llama-server}"
model_path="${MODEL_PATH:-}"
mtp_depth="${MTP_DEPTH:-4}"
threads="${THREADS:-6}"
batch_size="${BATCH_SIZE:-128}"
ubatch_size="${UBATCH_SIZE:-128}"
host="${HOST:-127.0.0.1}"
port="${PORT:-8080}"

if [[ -z "$model_path" ]]; then
    echo "Set MODEL_PATH to RVN-IQ3_XXS-multilingual-mtp.gguf." >&2
    exit 2
fi

case "$mtp_depth" in
    4|8) ;;
    *)
        echo "MTP_DEPTH must be 4 or 8; those are the validated presets." >&2
        exit 2
        ;;
esac

for value_name in batch_size ubatch_size; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer." >&2
        exit 2
    fi
done

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
    -m "$model_path" --alias "qwen3.8-27b-rvn-iq3-64k-depth${mtp_depth}" \
    -c 65536 --parallel 1 \
    -dev CUDA0 --fit off --n-gpu-layers 47 \
    --override-tensor 'blk\.(10|11|12|13|14|15|16)\..*=CUDA0' \
    --load-mode none \
    -ctk q4_0 -ctv q4_0 \
    -b "$batch_size" -ub "$ubatch_size" \
    -t "$threads" -tb "$threads" \
    --spec-type draft-mtp \
    --spec-draft-n-max "$mtp_depth" \
    --spec-draft-p-min 0 \
    --spec-draft-type-k q4_0 \
    --spec-draft-type-v q4_0 \
    --spec-draft-threads "$threads" \
    --spec-draft-threads-batch "$threads" \
    -fa on --no-mmproj --reasoning off --jinja \
    --cache-ram 0 --no-cache-idle-slots --no-ui \
    --host "$host" --port "$port" --metrics \
    "$@"
