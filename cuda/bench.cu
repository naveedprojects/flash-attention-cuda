#include "flash_attention_kernel.cuh"
#include <cstdio>

int main(int argc, char** argv) {
    int seq_len = argc > 1 ? atoi(argv[1]) : 1024;
    const int batch_size = 1, num_heads = 32, head_dim = 128;

    size_t sz = (size_t)batch_size * num_heads * seq_len * head_dim * sizeof(half);

    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, sz); cudaMalloc(&d_K, sz);
    cudaMalloc(&d_V, sz); cudaMalloc(&d_O, sz);

    // Initialize with random data
    half* h_data = (half*)malloc(sz);
    srand(42);
    for (size_t i = 0; i < sz/sizeof(half); i++) {
        h_data[i] = __float2half((float)rand()/RAND_MAX - 0.5f);
    }
    cudaMemcpy(d_Q, h_data, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_data, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_data, sz, cudaMemcpyHostToDevice);
    free(h_data);

    float scale = 1.0f / sqrtf(128.0f);
    dim3 grid((seq_len + 63) / 64, batch_size * num_heads);
    size_t smem = 32 * 136 * sizeof(half) * 2;

    cudaFuncSetAttribute(flash_attention_kernel<128, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    // Warmup
    for (int i = 0; i < 20; i++) {
        flash_attention_kernel<128, true><<<grid, 128, smem>>>(d_Q, d_K, d_V, d_O, seq_len, scale);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int iters = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        flash_attention_kernel<128, true><<<grid, 128, smem>>>(d_Q, d_K, d_V, d_O, seq_len, scale);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;

    long long flops = 4LL * batch_size * num_heads * seq_len * seq_len * head_dim / 2;
    float tflops = (flops / (ms / 1000.0f)) / 1e12f;
    printf("seq_len=%d: %.3f ms, %.2f TFLOPS (%.1f%% of PyTorch)\n",
           seq_len, ms, tflops, 100.0f * tflops / 21.88f);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}
