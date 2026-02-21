#include "flash_attention_kernel.cuh"
#include <cstdio>

int main(int argc, char** argv) {
    int seq_len = argc > 1 ? atoi(argv[1]) : 1024;
    const int batch_size = 1, num_heads = 32, head_dim = 128;

    size_t sz = (size_t)batch_size * num_heads * seq_len * head_dim * sizeof(half);

    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, sz); cudaMalloc(&d_K, sz);
    cudaMalloc(&d_V, sz); cudaMalloc(&d_O, sz);

    float scale = 1.0f / sqrtf(128.0f);
    dim3 grid((seq_len + 63) / 64, batch_size * num_heads);
    size_t smem = 32 * 136 * sizeof(half) * 2;

    cudaFuncSetAttribute(flash_attention_kernel<128, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    // Create CUDA stream for graph capture
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup before graph capture
    for (int i = 0; i < 10; i++) {
        flash_attention_kernel<128, true><<<grid, 128, smem, stream>>>(d_Q, d_K, d_V, d_O, seq_len, scale);
    }
    cudaStreamSynchronize(stream);

    // Capture graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    flash_attention_kernel<128, true><<<grid, 128, smem, stream>>>(d_Q, d_K, d_V, d_O, seq_len, scale);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Warmup graph execution
    for (int i = 0; i < 20; i++) {
        cudaGraphLaunch(graphExec, stream);
    }
    cudaStreamSynchronize(stream);

    // Benchmark with CUDA graph
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    const int iters = 100;

    // Regular launch benchmark
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; i++) {
        flash_attention_kernel<128, true><<<grid, 128, smem, stream>>>(d_Q, d_K, d_V, d_O, seq_len, scale);
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    float ms_regular;
    cudaEventElapsedTime(&ms_regular, start, stop);
    ms_regular /= iters;

    // Graph launch benchmark
    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; i++) {
        cudaGraphLaunch(graphExec, stream);
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    float ms_graph;
    cudaEventElapsedTime(&ms_graph, start, stop);
    ms_graph /= iters;

    long long flops = 4LL * batch_size * num_heads * seq_len * seq_len * head_dim / 2;
    float tflops_regular = (flops / (ms_regular / 1000.0f)) / 1e12f;
    float tflops_graph = (flops / (ms_graph / 1000.0f)) / 1e12f;

    printf("seq_len=%d:\n", seq_len);
    printf("  Regular: %.3f ms, %.2f TFLOPS (%.1f%% of PyTorch)\n",
           ms_regular, tflops_regular, 100.0f * tflops_regular / 21.88f);
    printf("  Graph:   %.3f ms, %.2f TFLOPS (%.1f%% of PyTorch)\n",
           ms_graph, tflops_graph, 100.0f * tflops_graph / 21.88f);
    printf("  Speedup: %.1f%%\n", 100.0f * (ms_regular - ms_graph) / ms_regular);

    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
    cudaStreamDestroy(stream);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}
