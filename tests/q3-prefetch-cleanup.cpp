#include "ggml-backend.cpp"
#include <cstdio>

static int released;
static void release_event(ggml_backend_dev_t, ggml_backend_event_t event) {
    ++released;
    delete event;
}

int main() {
    ggml_backend_device device = {};
    device.iface.event_free = release_event;
    for (int count = 0; count <= 3; ++count) {
        auto * sched = (ggml_backend_sched *) calloc(1, sizeof(ggml_backend_sched));
        sched->prefetch_failed = true;
        for (int slot = 0; slot < count; ++slot) {
            sched->queue_ready[slot] = new ggml_backend_event{};
            sched->queue_ready[slot]->device = &device;
        }
        released = 0;
        ggml_backend_sched_free(sched);
        if (released != count) {
            std::fprintf(stderr, "FAIL: backend absent, allocated=%d released=%d\n", count, released);
            return 1;
        }
    }
    std::puts("PASS: null backend cleanup releases zero through three partial events");
}
