// Compile-only toolkit/host-compiler compatibility probe. Never runs on a GPU.
__global__ void routeweaver_probe(float * value) { value[0] = 1.0f; }
