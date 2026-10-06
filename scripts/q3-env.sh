# Sourced by the Q3 launcher and focused regression runner.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
variant="${Q3_VARIANT:-candidate}"
case "$variant" in candidate|baseline) ;; *) echo 'Invalid Q3_VARIANT.' >&2; return 2 ;; esac
context="${CONTEXT:-98304}"
case "$context:$variant" in
    98304:candidate) default_blocks=36; default_output=32768 ;;
    131072:candidate) default_blocks=46; default_output=65536 ;;
    98304:baseline) default_blocks=44; default_output=32768 ;;
    131072:baseline) default_blocks=54; default_output=65536 ;;
    *) echo 'Supported Q3 presets: CONTEXT=98304 or 131072.' >&2; return 2 ;;
esac
host_blocks="${HOST_FFN_BLOCKS:-$default_blocks}"
threads="${THREADS:-6}"
port="${PORT:-8080}"
depth="${MTP_DEPTH:-4}"
speculation="${SPECULATION:-mtp}"
case "$speculation" in mtp|none) ;; *) echo 'SPECULATION must be mtp or none.' >&2; return 2 ;; esac
batch="${BATCH_SIZE:-128}"
ubatch="${UBATCH_SIZE:-128}"
output="${MAX_OUTPUT_TOKENS:-$default_output}"
for value in "$host_blocks" "$threads" "$port" "$depth" "$output" "$batch" "$ubatch"; do
    [[ "$value" =~ ^(0|[1-9][0-9]*)$ && ${#value} -le 6 ]] || { echo 'Invalid integer setting.' >&2; return 2; }
done
((host_blocks <= 64 && threads >= 1 && threads <= 256 && port >= 1 && port <= 65535 && depth >= 1 && depth <= 10 && output >= 1 && output < context)) || { echo 'Q3 setting outside supported bounds.' >&2; return 2; }
((ubatch >= 32 && ubatch <= batch && batch <= 512)) || { echo 'Require 32 <= UBATCH_SIZE <= BATCH_SIZE <= 512.' >&2; return 2; }
model="${MODEL_PATH:-$root/models/RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_XXS-mtp.gguf}"
template="${CHAT_TEMPLATE_PATH:-$root/models/qwen-fixed-v22.4.jinja}"
build_dir="$root/build/q3-$variant"
server="$build_dir/bin/llama-server"
tensor_pattern='blk\.('
for ((layer=0; layer<host_blocks; ++layer)); do
    [[ "$layer" == 0 ]] || tensor_pattern+='|'
    tensor_pattern+="$layer"
done
tensor_pattern+=')\.ffn_(gate|up)\.weight=CPU'
# Drop inherited experimental switches; all measured switches are explicit below.
while read -r variable; do
    case "$variable" in GGML_*|LLAMA_*) unset "$variable" ;; esac
done < <(compgen -e)
unset LD_PRELOAD
export LD_LIBRARY_PATH="$build_dir/bin${CUDA_TOOLKIT_ROOT:+:$CUDA_TOOLKIT_ROOT/lib64}"
export LLAMA_GDN_TRANSACTIONAL_REPLAY=1 LLAMA_GDN_COMPLETE_CHECKPOINTS=0
[[ "$variant" != candidate ]] || export LLAMA_GDN_COMPLETE_CHECKPOINTS=1
export LLAMA_COMPACT_GDN_ROLLBACK=0 LLAMA_PACKED_GDN_TREE=0
export GGML_CPU_IQ3XXS_MULTI=6 GGML_OP_OFFLOAD_MIN_BATCH=2
export GGML_CUDA_Q4_FA_HEADSPLIT=1 GGML_CUDA_Q4_FA_VEC=1 GGML_CUDA_Q4_FA_COLS=2
export GGML_CUDA_ASYNC_HOST_WEIGHTS=0 GGML_CUDA_WEIGHT_PREFETCH=1
export GGML_CUDA_WEIGHT_PREFETCH_DIRECT=0 GGML_CUDA_WEIGHT_PREFETCH_ALIAS=1
export GGML_CUDA_WEIGHT_PREFETCH_ALIAS_BOUNDS=1 GGML_CUDA_WEIGHT_PREFETCH_QUEUE=1
