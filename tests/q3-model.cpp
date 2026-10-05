#include "arg.h"
#include "common.h"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool b, const char * why) { if (!b) throw std::runtime_error(why); }
static void decode(llama_context * ctx, const std::vector<llama_token> & tokens, int start, int count) {
    auto batch = llama_batch_init(count, 0, 1);
    for (int i = 0; i < count; ++i) common_batch_add(batch, tokens.at(start+i), start+i, {0}, true);
    const int code = llama_decode(ctx, batch);
    llama_batch_free(batch);
    require(code == 0, "decode failed");
}
int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    ggml_backend_load_all();
    auto init = common_init_from_params(params);
    auto * model = init->model();
    require(model, "model not loaded");
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto tokens = common_tokenize(init->context(),
        "Write a function to parse a signed integer, rejecting invalid text and detecting overflow. "
        "Then design tests for leading whitespace, trailing characters, empty strings, and limits.", true);
    const std::vector<llama_token> padding(tokens.begin(), tokens.begin() + 8);
    while (tokens.size() < 128) tokens.insert(tokens.end(), padding.begin(), padding.end());
    try {
        for (unsigned depth : {4u, 5u, 8u}) {
            auto config = common_context_params_to_llama(params);
            config.n_ctx = 256; config.n_seq_max = 1; config.n_rs_seq = depth;
            config.n_batch = 128; config.n_ubatch = 128;
            auto * src = llama_init_from_model(model, config);
            auto * dst = llama_init_from_model(model, config);
            require(src && dst, "context allocation failed");
            for (unsigned rollback = 0; rollback <= depth; ++rollback) {
                llama_memory_clear(llama_get_memory(src), true);
                llama_memory_clear(llama_get_memory(dst), true);
                decode(src, tokens, 0, 17);
                decode(src, tokens, 17, depth + 1);
                const int resume = 18 + depth - rollback;
                if (rollback) require(llama_memory_seq_rm(llama_get_memory(src), 0, resume, -1), "rollback failed");
                common_prompt_checkpoint checkpoint;
                checkpoint.update_tgt(src, 0, 0);
                // Restore into an already-used context, then compare every output logit.
                decode(dst, tokens, 0, 11);
                checkpoint.load_tgt(dst, 0, 0);
                decode(src, tokens, resume, 1);
                decode(dst, tokens, resume, 1);
                const float * a = llama_get_logits_ith(src, 0);
                const float * b = llama_get_logits_ith(dst, 0);
                float max_diff = 0;
                for (int i = 0; i < vocab; ++i) {
                    require(std::isfinite(a[i]) && std::isfinite(b[i]), "non-finite logit");
                    max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
                }
                printf("depth=%u rollback=%u restored-logit-max-diff=%g\n", depth, rollback, (double) max_diff);
                fflush(stdout);
                require(max_diff <= 1e-5f, "restored continuation differs from uninterrupted continuation");
            }
            llama_free(dst); llama_free(src);
        }
        puts("PASS: all real-model pending replay snapshots preserve continuation logits");
    } catch (const std::exception & e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
