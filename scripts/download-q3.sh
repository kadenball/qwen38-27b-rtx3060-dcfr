#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
directory="${MODEL_DIR:-$root/models}"
[[ $# == 0 || ( $# == 1 && ( "$1" == --template-only || "$1" == --vision-only ) ) ]] || { echo 'Usage: download-q3.sh [--template-only|--vision-only]' >&2; exit 2; }
mkdir -p "$directory"
command -v flock >/dev/null || { echo 'Install util-linux (flock) first.' >&2; exit 2; }
exec 9>"$directory/.routeweaver-download.lock"
flock -n 9 || { echo 'A RouteWeaver download is already running.' >&2; exit 1; }
download() {
    local name="$1" digest="$2" url="$3"
    if [[ -e "$directory/$name" ]]; then
        printf '%s  %s\n' "$digest" "$directory/$name" | sha256sum -c -
        return
    fi
    curl -fL --retry 3 --connect-timeout 30 --continue-at - "$url" -o "$directory/$name.partial"
    printf '%s  %s\n' "$digest" "$directory/$name.partial" | sha256sum -c -
    mv -n "$directory/$name.partial" "$directory/$name"
    printf '%s  %s\n' "$digest" "$directory/$name" | sha256sum -c -
}
if [[ "${1:-}" == --vision-only ]]; then
    download mmproj-F16.gguf cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e \
        https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4ca720788d1e01f1bff70c033e0d0028fd02e502/mmproj-F16.gguf
    exit 0
fi
download qwen-fixed-v22.4.jinja c47c82b0544752d454f4e427228d9d9d8c3df64c9e446cbd0229362f67948009 \
    https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/756cfb69d742355fd310b4ba9d50815a27d9d241/chat_template.jinja
[[ "${1:-}" != --template-only ]] || exit 0
download RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_XXS-mtp.gguf 475be499f4bc4f729a811e24ad11419bd187d345af18eff494b89c8ba55a0039 \
    https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-GSQ-RCO-GGUF/resolve/1b65f1eb296b08113e89b43d8ab634c6b555bf98/RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_XXS-mtp.gguf
