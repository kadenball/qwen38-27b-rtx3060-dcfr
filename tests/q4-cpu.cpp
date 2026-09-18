#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "quants.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

extern "C" void ggml_vec_dot_iq4_xs_q8_K_multi(int n, float * s, size_t bs, const void * vx, const void * vy, size_t by, int nq);

static void fill_x(void * data, int blocks, std::mt19937 & rng) {
    auto * x = static_cast<block_iq4_xs *>(data);
    for (int b = 0; b < blocks; ++b) {
        x[b].d = ggml_fp32_to_fp16((int(rng() % 2001) - 1000) * 0.00001f);
        x[b].scales_h = rng();
        for (auto & s : x[b].scales_l) s = rng();
        for (auto & q : x[b].qs) q = rng();
    }
}

static void fill_y(void * data, int blocks, std::mt19937 & rng) {
    auto * y = static_cast<block_q8_K *>(data);
    for (int b = 0; b < blocks; ++b) {
        y[b].d = (1 + rng() % 1000) * 0.00001f;
        for (auto & q : y[b].qs) q = int(rng() % 255) - 127;
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int k = 0; k < 16; ++k) sum += y[b].qs[j * 16 + k];
            y[b].bsums[j] = sum;
        }
    }
}

static bool check() {
    std::mt19937 rng(20260914);
    int cases = 0;
    for (int n : {256, 512, 5120, 17408}) {
        for (int padding : {0, 64}) {
            const size_t bx = n / QK_K * sizeof(block_iq4_xs) + padding;
            const size_t by = n / QK_K * sizeof(block_q8_K) + padding;
            std::vector<unsigned char> x(2 * bx), y(6 * by);
            for (int trial = 0; trial < 32; ++trial) {
              for (int nq = 1; nq <= 6; ++nq) {
                for (int r = 0; r < 2; ++r) {
                    fill_x(x.data() + r * bx, n / QK_K, rng);
                }
                for (int q = 0; q < nq; ++q) fill_y(y.data() + q * by, n / QK_K, rng);
                float ref[6];
                float out[256];
                for (auto & v : out) v = 123456.0f;
                const size_t bs = trial % 2 ? 16 : 32;
                for (int q = 0; q < nq; ++q) {
                    ggml_vec_dot_iq4_xs_q8_K(n, &ref[q], 0,
                        x.data(), 0, y.data() + q * by, 0, 1);
                }
                ggml_vec_dot_iq4_xs_q8_K_multi(n, out, bs, x.data(), y.data(), by, nq);
                for (size_t i = 0; i < 256; ++i) {
                    const bool written = i % bs == 0 && i / bs < size_t(nq);
                    const float expected = written ? ref[i / bs] : 123456.0f;
                    if (std::memcmp(&out[i], &expected, sizeof(float)) != 0) {
                        std::cerr << "FAIL n=" << n << " padding=" << padding << " trial=" << trial
                                  << " nq=" << nq << " output=" << i << " expected=" << expected << " actual=" << out[i] << '\n';
                        return false;
                    }
                }
                ++cases;
              }
            }
        }
    }
    std::cout << "PASS " << cases << " exact multi-query cases, including output sentinels\n";
    return true;
}

static void bench() {
    std::mt19937 rng(20260914);
    for (int n : {5120, 17408}) {
        const int rows = n == 5120 ? 17408 : 5120;
        const size_t bx = n / QK_K * sizeof(block_iq4_xs);
        const size_t by = n / QK_K * sizeof(block_q8_K);
        std::vector<unsigned char> x(rows * bx), y(2 * by);
        std::vector<float> out(rows * 2);
        fill_x(x.data(), rows * n / QK_K, rng);
        fill_y(y.data(), 2 * n / QK_K, rng);
        for (int round = 0; round < 6; ++round) {
            for (int phase = 0; phase < 2; ++phase) {
                const int mode = (round + phase) % 2;
                const auto start = std::chrono::steady_clock::now();
                for (int r = 0; r < rows; r += 2) {
                    if (mode == 0) {
                        for (int q = 0; q < 2; ++q) {
                            for (int rr = 0; rr < 2; ++rr) {
                                ggml_vec_dot_iq4_xs_q8_K(n, &out[q * rows + r + rr], 0,
                                    x.data() + (r + rr) * bx, 0, y.data() + q * by, 0, 1);
                            }
                        }
                    } else {
                        ggml_vec_dot_iq4_xs_q8_K(n, &out[r], rows,
                            x.data() + r * bx, bx, y.data(), by, 2);
                    }
                }
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                std::cout << "n=" << n << " rows=" << rows << " round=" << round
                          << " mode=" << mode << " ms=" << ms << " check=" << out.back() << '\n';
            }
        }
    }
}

static bool check_graphs() {
    std::mt19937 rng(20260914);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 6);
    int cases = 0;
    for (int n : {256, 5120, 17408}) {
        for (int rows : {4, 5, n == 5120 ? 17408 : 5120}) {
            for (int batch : {1, 2, 3, 4, 5, 6, 7, 8, 10, 16}) {
                for (bool strided : {false, true}) {
                    if (rows > 5 && (n == 256 || batch > 8 || strided)) continue;
                    ggml_init_params params = {2 << 20, nullptr, true};
                    ggml_context * ctx = ggml_init(params);
                    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, n, rows);
                    ggml_tensor * ybase = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, batch * (strided ? 2 : 1));
                    ggml_tensor * y = strided
                        ? ggml_view_2d(ctx, ybase, n, batch, ybase->nb[1] * 2, 0) : ybase;
                    ggml_tensor * out = ggml_mul_mat(ctx, x, y);
                    ggml_cgraph * graph = ggml_new_graph(ctx);
                    ggml_build_forward_expand(graph, out);
                    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                    if (!buffer) return false;
                    ggml_backend_buffer_clear(buffer, 0);
                    std::vector<unsigned char> weights(ggml_nbytes(x));
                    fill_x(weights.data(), rows * n / QK_K, rng);
                    ggml_backend_tensor_set(x, weights.data(), 0, weights.size());
                    std::vector<float> queries(ggml_nelements(ybase));
                    for (auto & v : queries) v = (int(rng() % 2001) - 1000) * 0.001f;
                    ggml_backend_tensor_set(ybase, queries.data(), 0, ggml_nbytes(ybase));
                    std::vector<float> ref(rows * batch), actual(rows * batch);
                    ggml_backend_cpu_set_use_ref(backend, true);
                    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
                    ggml_backend_tensor_get(out, ref.data(), 0, ggml_nbytes(out));
                    ggml_backend_cpu_set_use_ref(backend, false);
                    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
                    ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
                    for (size_t i = 0; i < ref.size(); ++i) {
                        if (!std::isfinite(ref[i]) || !std::isfinite(actual[i]) ||
                            std::memcmp(&ref[i], &actual[i], sizeof(float)) != 0) {
                            std::cerr << "GRAPH FAIL n=" << n << " rows=" << rows << " batch=" << batch
                                      << " strided=" << strided << " index=" << i
                                      << " ref=" << ref[i] << " actual=" << actual[i] << '\n';
                            return false;
                        }
                    }
                    ggml_backend_buffer_free(buffer);
                    ggml_free(ctx);
                    ++cases;
                }
            }
        }
    }
    ggml_backend_free(backend);
    std::cout << "PASS " << cases << " exact initialized MUL_MAT graphs at six threads\n";
    return true;
}

int main(int argc, char ** argv) {
    ggml_cpu_init();
    if (!check()) return 1;
    if (argc == 2 && std::string(argv[1]) == "--bench") bench();
    if (argc == 2 && std::string(argv[1]) == "--graphs" && !check_graphs()) return 1;
}
