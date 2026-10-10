#include "sampling.cpp"
#include <array>
#include <iostream>
#include <numeric>
#include <stdexcept>

static std::vector<float> test_logits;
// Fake logits only; all sampling and rejection code is the actual implementation.
extern "C" {
const llama_model * llama_get_model(const llama_context *) { return nullptr; }
const llama_vocab * llama_model_get_vocab(const llama_model *) { return nullptr; }
int32_t llama_vocab_n_tokens(const llama_vocab *) { return test_logits.size(); }
void llama_synchronize(llama_context *) {}
float * llama_get_logits_ith(llama_context *, int32_t) { return test_logits.data(); }
llama_token llama_get_sampled_token_ith(llama_context *, int32_t) { return LLAMA_TOKEN_NULL; }
float * llama_get_sampled_probs_ith(llama_context *, int32_t) { return nullptr; }
uint32_t llama_get_sampled_probs_count_ith(llama_context *, int32_t) { return 0; }
float * llama_get_sampled_logits_ith(llama_context *, int32_t) { return nullptr; }
uint32_t llama_get_sampled_logits_count_ith(llama_context *, int32_t) { return 0; }
llama_token * llama_get_sampled_candidates_ith(llama_context *, int32_t) { return nullptr; }
}

static common_sampler_ptr make_sampler() {
    common_params_sampling params;
    params.seed = 731;
    params.no_perf = true;
    auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_dist(params.seed));
    return common_sampler_ptr(new common_sampler {
        params, nullptr, nullptr, chain, ring_buffer<llama_token>(64), {}, {},
        std::mt19937(params.seed ^ 0x9e3779b9u),
    });
}

static void require(bool ok, const char * label) {
    if (!ok) throw std::runtime_error(label);
}

static void check(const std::vector<float> & p, const std::vector<float> & q) {
    test_logits.clear();
    for (float x : p) test_logits.push_back(x > 0 ? std::log(x) : -INFINITY);
    std::vector<llama_token_data> q_data;
    for (size_t i = 0; i < q.size(); ++i) if (q[i] > 0) q_data.push_back({(llama_token)i, 0, q[i]});
    auto sampler = make_sampler();
    auto * ctx = reinterpret_cast<llama_context *>(uintptr_t(1));
    std::mt19937 draft_rng(9223);
    std::discrete_distribution<int> draw(q.begin(), q.end());
    constexpr int trials = 200000;
    std::vector<int> counts(p.size(), 0);
    int accepted = 0;
    for (int n = 0; n < trials; ++n) {
        const auto result = common_sampler_sample_and_accept_n_rejection(
                sampler.get(), ctx, {0,1}, {draw(draft_rng)}, {q_data});
        ++counts.at(result.at(0));
        accepted += result.size() == 2;
    }
    double overlap = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        require(std::abs(double(counts[i])/trials - p[i]) < 0.006, "target marginal changed");
        overlap += std::min(p[i], q[i]);
    }
    require(std::abs(double(accepted)/trials-overlap)<0.006, "acceptance differs from overlap");

    const llama_token draft_id = q_data.front().id;
    common_sampler_reset(sampler.get());
    const auto once = common_sampler_sample_and_accept_n_rejection(sampler.get(),ctx,{0,1},{draft_id},{q_data});
    common_sampler_reset(sampler.get());
    require(once == common_sampler_sample_and_accept_n_rejection(sampler.get(),ctx,{0,1},{draft_id},{q_data}), "reset changes stream");
    common_sampler_ptr clone(common_sampler_clone(sampler.get()));
    const auto a = common_sampler_sample_and_accept_n_rejection(sampler.get(),ctx,{0,1},{draft_id},{q_data});
    const auto b = common_sampler_sample_and_accept_n_rejection(clone.get(),ctx,{0,1},{draft_id},{q_data});
    require(a == b, "clone changes stream");
    common_sampler_copy(sampler.get(),clone.get());
    require(common_sampler_sample_and_accept_n_rejection(sampler.get(),ctx,{0,1},{draft_id},{q_data}) ==
            common_sampler_sample_and_accept_n_rejection(clone.get(),ctx,{0,1},{draft_id},{q_data}), "copy changes stream");

    std::vector<int> joint(p.size()*p.size(), 0);
    for (int n = 0; n < trials; ++n) {
        llama_tokens sequence;
        while (sequence.size() < 2) {
            const auto batch = common_sampler_sample_and_accept_n_rejection(
                    sampler.get(),ctx,{0,1,2},{draw(draft_rng),draw(draft_rng)},{q_data,q_data});
            sequence.insert(sequence.end(),batch.begin(),batch.end());
        }
        ++joint.at(sequence[0]*p.size()+sequence[1]);
    }
    for(size_t i=0;i<p.size();++i)for(size_t j=0;j<p.size();++j) {
        require(std::abs(double(joint[i*p.size()+j])/trials-p[i]*p[j])<0.006, "two-token joint law changed");
    }
    std::cout << "PASS target law, acceptance, reset, clone, copy; overlap=" << overlap << '\n';
}

int main() {
    check({0.2f,0.3f,0.5f}, {0.2f,0.3f,0.5f});
    check({0.8f,0.1f,0.1f}, {0.2f,0.5f,0.3f});
    check({0,0.3f,0.7f}, {0.4f,0.6f,0});
    check({0.2f,0.3f,0.5f}, {1,0,0});
    check({0,0,1}, {1,0,0});
    check({0,1,0}, {0,1,0});
    std::cout << "All 1,200,000 marginal and 1,200,000 two-token sequence trials passed.\n";
}
