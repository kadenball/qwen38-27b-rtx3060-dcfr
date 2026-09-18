#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${MODEL_PATH:-$root/models/Qwen3.8-27B-IQ4_XS.gguf}"
server="$root/build/q4/bin/llama-server"
[[ -f "$model" && -x "$server" ]] || { echo 'Build Q4 and download/verify the model first. See docs/q4-quickstart.md.' >&2; exit 2; }
# Pin measured policy; do not inherit unrelated experimental settings.
export LLAMA_GDN_TRANSACTIONAL_REPLAY=1 LLAMA_COMPACT_GDN_ROLLBACK=0
export LLAMA_PACKED_GDN_TREE=0 GGML_CUDA_ASYNC_HOST_WEIGHTS=0
export GGML_CUDA_Q4_FA_HEADSPLIT=1 GGML_CUDA_Q4_FA_VEC=1 GGML_CUDA_Q4_FA_COLS=2
export GGML_CPU_IQ4XS_DOT2=1 GGML_CPU_IQ4XS_MULTI=6 GGML_OP_OFFLOAD_MIN_BATCH=32
# Use this build's libraries, not an unrelated router's injected libraries.
export LD_LIBRARY_PATH="$root/build/q4/bin${CUDA_TOOLKIT_ROOT:+:$CUDA_TOOLKIT_ROOT/lib64}"
exec "$server" -m "$model" --alias qwen38-q4-32k \
    --host 127.0.0.1 --port "${PORT:-8080}" --parallel 1 \
    --ctx-size 32768 --ctx-checkpoints 4 --load-mode mmap \
    --n-gpu-layers all --fit off --device CUDA0 \
    --override-tensor 'blk\.([0-9]|[123][0-9]|4[0-2])\.ffn_(gate|up|down)\.weight=CPU' \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 \
    --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
    --threads 6 --threads-batch 6 --spec-draft-threads 6 --spec-draft-threads-batch 6 \
    --batch-size 128 --ubatch-size 128 --flash-attn on \
    --cache-ram 0 --no-cache-idle-slots --no-mmproj --jinja \
    --reasoning off --chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":true}' \
    --no-ui --metrics "$@"
