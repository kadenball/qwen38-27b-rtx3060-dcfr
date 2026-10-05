#include "llama-model.h"
#include "llama-memory-recurrent.h"
#include "llama-io.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

struct writer : llama_io_write_i {
    std::vector<unsigned char> bytes;
    void write(const void * p, size_t n) override {
        auto b = static_cast<const unsigned char *>(p);
        bytes.insert(bytes.end(), b, b + n);
    }
    void write_tensor(ggml_tensor * t, size_t offset, size_t n) override {
        if (offset + n > ggml_nbytes(t)) throw std::runtime_error("checkpoint reads outside tensor");
        std::vector<unsigned char> b(n);
        ggml_backend_tensor_get(t, b.data(), offset, n);
        write(b.data(), n);
    }
    size_t n_bytes() override { return bytes.size(); }
};
struct reader : llama_io_read_i {
    const std::vector<unsigned char> & bytes;
    size_t pos = 0;
    reader(const writer & w) : bytes(w.bytes) {}
    void read(void * p, size_t n) override {
        if (pos + n > bytes.size()) throw std::runtime_error("truncated checkpoint");
        memcpy(p, bytes.data() + pos, n); pos += n;
    }
    void read_tensor(ggml_tensor * t, size_t offset, size_t n) override {
        if (offset + n > ggml_nbytes(t)) throw std::runtime_error("checkpoint writes outside tensor");
        std::vector<unsigned char> b(n); read(b.data(), n);
        ggml_backend_tensor_set(t, b.data(), offset, n);
    }
    size_t n_bytes() override { return pos; }
};
static void require(bool value, const char * why) {
    if (!value) throw std::runtime_error(why);
}
int main() {
    setenv("LLAMA_GDN_TRANSACTIONAL_REPLAY", "1", 1);
    ggml_backend_load_all();
    std::unique_ptr<llama_model> owned(llama_model_create(LLM_ARCH_QWEN35, llama_model_default_params()));
    auto & model = *owned;
    model.arch = LLM_ARCH_QWEN35;
    model.hparams.n_layer_all = 2;
    model.hparams.n_embd = 8;
    model.hparams.ssm_d_conv = 4;
    model.hparams.ssm_d_state = 4;
    model.hparams.ssm_d_inner = 8;
    model.hparams.ssm_dt_rank = 2;
    model.hparams.ssm_n_group = 1;
    try {
        for (unsigned rollback = 0; rollback < 5; ++rollback) {
            llama_memory_recurrent src(model, GGML_TYPE_F32, GGML_TYPE_F32, false, 2, 2, 5, {});
            llama_memory_recurrent dst(model, GGML_TYPE_F32, GGML_TYPE_F32, false, 2, 2, 5, {});
            src.cells[0].pos = 20 - rollback;
            src.cells[0].src = 0; src.cells[0].tail = 0;
            src.cells[0].seq_id.insert(0); src.used = 1;
            src.rs_last_n[0] = 5; src.rs_idx[0] = rollback;
            for (auto group : {src.r_l, src.s_l, src.f_l}) for (auto * t : group) {
                std::vector<float> data(ggml_nelements(t));
                for (size_t i = 0; i < data.size(); ++i) data[i] = (int(i % 97) - 48) * 0.03125f;
                ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
            }
            writer w; src.state_write(w, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            reader r(w); dst.state_read(r, 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            require(r.n_bytes() == w.n_bytes(), "checkpoint not consumed");
            require(dst.rs_last_n[1] == 5, "FAIL: pending replay count lost on restore");
            require(dst.rs_idx[1] == rollback, "FAIL: rollback index lost on restore");
            const int target = dst.cells[1].tail;
            require(target >= 0, "restored sequence missing");
            for (unsigned group = 0; group < 3; ++group) {
                const auto & a = group == 0 ? src.r_l : group == 1 ? src.s_l : src.f_l;
                const auto & b = group == 0 ? dst.r_l : group == 1 ? dst.s_l : dst.f_l;
                for (size_t layer = 0; layer < a.size(); ++layer) {
                    const size_t row = a[layer]->nb[1];
                    for (int64_t plane = 0; plane < a[layer]->ne[1] / src.size; ++plane) {
                        std::vector<unsigned char> x(row), y(row);
                        ggml_backend_tensor_get(a[layer], x.data(), plane * src.size * row, row);
                        ggml_backend_tensor_get(b[layer], y.data(), (plane * dst.size + target) * row, row);
                        require(x == y, "FAIL: recurrent snapshot/factors differ");
                    }
                }
            }
            // Whole-cache snapshots retain shared sequence ownership and replay metadata.
            src.cells[0].seq_id.insert(1); src.cells[1].tail = 0;
            src.rs_idx[1] = rollback; src.rs_last_n[1] = 5;
            writer all; src.state_write(all, -1, 0);
            reader all_read(all); dst.state_read(all_read, -1, 0);
            require(dst.cells[0].seq_id == src.cells[0].seq_id, "shared owners lost");
            require(dst.rs_idx[1] == rollback && dst.rs_last_n[1] == 5, "shared replay metadata lost");
            writer roundtrip; dst.state_write(roundtrip, -1, 0);
            require(roundtrip.bytes == all.bytes, "whole-cache roundtrip differs");

            writer truncated = w; truncated.bytes.resize(truncated.bytes.size() - 16);
            reader bad(truncated); bool rejected = false;
            try { dst.state_read(bad, 1, 0); } catch (const std::exception &) { rejected = true; }
            require(rejected && dst.seq_pos_max(1) == -1, "truncated checkpoint did not clear destination");
            reader recover(w); dst.state_read(recover, 1, 0);
            require(dst.rs_last_n[1] == 5, "valid checkpoint failed after bad input");

            src.rs_last_n[1] = 4; rejected = false;
            try { writer invalid; src.state_write(invalid, -1, 0); } catch (const std::exception &) { rejected = true; }
            require(rejected, "inconsistent shared replay metadata accepted");
        }
        for (uint32_t last : {2u, 17u}) {
            llama_memory_recurrent mem(model, GGML_TYPE_F32, GGML_TYPE_F32, false, 2, 2, 5, {});
            mem.cells[0].pos = 20; mem.cells[0].src = 0; mem.cells[0].tail = 0;
            mem.cells[0].seq_id.insert(0); mem.used = 1; mem.rs_last_n[0] = last;
            require(!mem.seq_rm(0, last == 2 ? 18 : 20, -1), "FAIL: rollback accepted without retained factors");
        }
        puts("PASS: five exact snapshots, whole/shared cache, truncated input recovery, rollback bounds");
    } catch (const std::exception & e) { fprintf(stderr, "%s\n", e.what()); return 1; }
}
