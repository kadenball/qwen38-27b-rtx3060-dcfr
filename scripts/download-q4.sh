#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
directory="${MODEL_DIR:-$root/models}"
name=Qwen3.8-27B-IQ4_XS.gguf
revision=125a02af4987b57c7deb88d7f2ec58a5725c07c0
digest=4b927360fa7c4aa41a734302f0795ca45df0e8b8a258bc76cc5fbf9d98a484db
mkdir -p "$directory"
verify() { echo "$digest  $1" | sha256sum -c -; }
if [[ -e "$directory/$name" ]]; then
    verify "$directory/$name"
    exit
fi
curl -fL --retry 3 --continue-at - \
    "https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/resolve/$revision/$name" \
    -o "$directory/$name.partial"
verify "$directory/$name.partial"
mv -n "$directory/$name.partial" "$directory/$name"
verify "$directory/$name"
