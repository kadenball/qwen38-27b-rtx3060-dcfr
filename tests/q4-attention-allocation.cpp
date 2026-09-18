#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Expected vector, mma, or headsplit allocation contract\n");
        return 2;
    }
    const std::string mode = argv[1];
    if (mode != "vector" && mode != "mma" && mode != "headsplit") return 2;
    // Select the expected contract separately from the actual environment.
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 2;
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(0);
    int checks = 0;
    int failed = 0;
    std::puts("layout,kv,queries,guard,output_bytes,allocation_bytes,expected_bytes");
    for (bool strided : {false, true}) {
        for (int64_t kv : {4096, 131072}) {
            for (int64_t queries : {1, 2, 6, 128, 129}) {
                for (int guard = 0; guard < 4; ++guard) {
                    ggml_init_params params = {1 << 20, nullptr, true};
                    ggml_context * ctx = ggml_init(params);
                    ggml_tensor * q = strided
                        ? ggml_permute(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 24, queries, 1), 0, 2, 1, 3)
                        : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, queries, 24, 1);
                    const int64_t capacity = strided ? 131072 : kv;
                    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, 256, capacity, 4, 1);
                    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, 256, capacity, 4, 1);
                    if (strided) {
                        k = ggml_view_4d(ctx, k, 256, kv, 4, 1, k->nb[1], k->nb[2], k->nb[3], 0);
                        v = ggml_view_4d(ctx, v, 256, kv, 4, 1, v->nb[1], v->nb[2], v->nb[3], 0);
                    }
                    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, queries, 1, 1);
                    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.0625f,
                        guard == 1 ? 1.0f : 0.0f, guard == 2 ? 50.0f : 0.0f);
                    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
                    if (guard == 3) ggml_flash_attn_ext_add_sinks(out, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 24));
                    const size_t payload = ggml_nbytes(out);
                    size_t expected = payload;
                    if (mode == "headsplit" && queries >= 2 && queries <= 128 && guard == 0) {
                        expected += 256*6*queries*sizeof(float) + 2*256*kv*sizeof(ggml_fp16_t);
                    } else if (queries != 1 && (mode == "mma" || queries > 128)) {
                        expected += 2*256*kv*4*sizeof(ggml_fp16_t);
                    }
                    const size_t actual = ggml_backend_buft_get_alloc_size(buft, out);
                    std::printf("%s,%lld,%lld,%d,%zu,%zu,%zu\n", strided ? "strided" : "plain",
                        (long long) kv, (long long) queries, guard, payload, actual, expected);
                    ++checks;
                    if (actual != expected) ++failed;
                    ggml_free(ctx);
                }
            }
        }
    }
    ggml_backend_free(backend);
    std::fprintf(stderr, "%d/%d allocation contracts passed\n", checks - failed, checks);
    return failed ? 1 : 0;
}
