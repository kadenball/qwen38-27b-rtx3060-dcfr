# Provenance and scope

The research patch in this repository applies to
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) revision
`c060ca974c773c7c3d17fd1b66dc9d312bc292c0`, which is distributed under the
MIT License. The patch is provided as source differences against that pinned
revision and retains upstream notices in the files it modifies.

The runtime expects a separately obtained GGUF distributed under its
applicable model license. The original August benchmarked file was
`Qwen3.8-27B-UD-IQ3_XXS.gguf`, SHA-256
`c0b7c3038681ed2e3040456c1dd45f9858b6c2290bed172c70388a94874f3eee`.

The patch is an experimental research snapshot, not an upstream llama.cpp
release. It contains D-CFR and the supporting recurrent-state, CUDA,
speculative-server, profiling, packed-tree, and branch-verification work that
was present in the measured runtime. Optional research paths are disabled
unless their environment flags are explicitly enabled.

The Q4 interactive overlay applies after that historical patch to the same
upstream base. It includes the measured head-split CUDA attention workspace,
AVX2 IQ4_XS dot2/multi-query CPU changes, and shallow-depth prompt-cache policy.
The optional DSH patches modify MIT-licensed `@deepseek-ai/dsh-compaction-basic`
and `@earendil-works/pi-ai` distributions; their upstream notices remain in the
installed packages. No model weights or third-party compiled binaries are
distributed here. The separate Q4 guide pins the Bartowski model revision and
SHA-256; model license terms remain those of its publisher.

The October 5 Q3 overlays add the IQ3_XXS multi-query kernel, queued transfer
scheduler, complete recurrent checkpoint codec and 48 MiB staging regions.
They apply after the existing research/Q4 overlays to the same pinned upstream
base. D-CFR denotes Deferred-Commit Factor Replay; RouteWeaver is the project
brand, not an upstream endorsement or a new model architecture.

The current model is `RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_XXS-mtp.gguf`, SHA-256
`475be499f4bc4f729a811e24ad11419bd187d345af18eff494b89c8ba55a0039`, distributed by
[0bserverx](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-GSQ-RCO-GGUF).
Credit for the model, fine-tuning and quantization remains with the respective
publishers; this project measures runtime improvements on that artifact.
The downloader separately retrieves the Apache-2.0-declared model and
froggeric's fixed chat template, plus an optional Unsloth-distributed projector.
They are not bundled with this source repository.

The banner is generated artwork created with the built-in image-generation
tool. Its exact prompt is retained in `docs/assets/banner-prompt.txt`.
The runtime research and release preparation were developed with AI assistance;
the included tests document the tested scope rather than guaranteeing correctness
over every workload or device.
