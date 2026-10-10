#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/q3-env.sh"
alias="routeweaver-q3-$context"
[[ "$variant" != fast ]] || alias="routeweaver-q3-fast-$context"
args=(-m "$model" --alias "$alias" --host 127.0.0.1 --port "$port"
    --parallel 1 --ctx-size "$context" --ctx-checkpoints 4 --load-mode none
    --n-gpu-layers all --fit off --device CUDA0
    --cache-type-k q4_0 --cache-type-v q4_0
    --threads "$threads" --threads-batch "$threads" --spec-draft-threads "$threads" --spec-draft-threads-batch "$threads"
    --batch-size "$batch" --ubatch-size "$ubatch" --flash-attn on
    --cache-ram 0 --no-cache-idle-slots --jinja --chat-template-file "$template"
    --reasoning on --chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}'
    --temp 1 --top-p 0.95 --top-k 20 --min-p 0 --n-predict "$output" --no-ui --metrics)
if [[ "$speculation" == mtp ]]; then
    args+=(--spec-type draft-mtp --spec-draft-n-max "$depth" --spec-draft-p-min 0 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0)
    [[ "$variant" != fast ]] || args+=(--spec-draft-sampling probabilistic)
else
    args+=(--spec-type none)
fi
((host_blocks == 0)) || args+=(--override-tensor "$tensor_pattern")
if [[ -n "${MMPROJ_PATH:-}" ]]; then args+=(--mmproj "$MMPROJ_PATH" --no-mmproj-offload)
else args+=(--no-mmproj); fi
if [[ "${1:-}" == --print-config ]]; then
    printf '%q ' "$server" "${args[@]}"; printf '\n'; exit 0
fi
[[ $# == 0 ]] || { echo 'Use documented environment settings; arbitrary duplicate flags are not accepted.' >&2; exit 2; }
[[ -x "$server" && -f "$model" && -f "$template" ]] || { echo 'Build Q3 and download the model/template first. See docs/q3-quickstart.md.' >&2; exit 2; }
[[ -z "${MMPROJ_PATH:-}" || -f "$MMPROJ_PATH" ]] || { echo 'MMPROJ_PATH does not exist.' >&2; exit 2; }
printf '%s  %s\n' c47c82b0544752d454f4e427228d9d9d8c3df64c9e446cbd0229362f67948009 "$template" | sha256sum -c -
printf '%s  %s\n' 475be499f4bc4f729a811e24ad11419bd187d345af18eff494b89c8ba55a0039 "$model" | sha256sum -c -
if [[ -n "${MMPROJ_PATH:-}" ]]; then
    printf '%s  %s\n' cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e "$MMPROJ_PATH" | sha256sum -c -
fi
echo "RouteWeaver Q3: $variant, context $context, CPU gate/up blocks $host_blocks, speculation $speculation, depth $depth."
exec "$server" "${args[@]}"
