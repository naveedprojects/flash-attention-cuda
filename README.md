# Flash Attention CUDA

A high-performance Flash Attention implementation using CUDA tensor cores that **beats PyTorch's cuDNN implementation** at longer sequence lengths.

## Performance Results

On RTX 3080 Ti Laptop GPU (batch=1, heads=32, head_dim=128, causal=true):

| Sequence Length | TFLOPS | vs PyTorch (21.88 TFLOPS) |
|-----------------|--------|---------------------------|
| 512             | 13.9   | 63%                       |
| 1024            | 19.4   | 89%                       |
| 2048            | 20.9   | 95%                       |
| 4096            | **22.6** | **103%** (beats PyTorch!) |
| 8192            | **22.5** | **103%** (beats PyTorch!) |

## Features

- **Adaptive Tile Size Dispatcher**: Automatically selects optimal kernel based on sequence length
- **Tensor Core Acceleration**: Uses `mma.sync.aligned.m16n8k16` PTX instructions
- **Two Optimized Kernels**:
  - Small tile (32x32, 64 threads) for seq_len < 768
  - Large tile (64x32, 128 threads) for seq_len >= 768
- **Online Softmax**: Numerically stable with warp-level shuffle reductions
- **Fast Exponential**: `exp2f(x * LOG2E)` instead of `expf(x)`
- **Causal Masking**: Built-in support for autoregressive attention
- **Memory Efficient**: O(N) memory instead of O(N^2) by tiling

## Requirements

- NVIDIA GPU with compute capability 8.0+ (Ampere or newer)
- CUDA Toolkit 11.0+
- C++17 compiler

## Build

```bash
# Using make (default: sm_86 for RTX 3080 Ti)
make

# For different GPU architecture
make ARCH=sm_80  # A100
make ARCH=sm_89  # RTX 4090
```

Or compile directly:
```bash
nvcc -O3 -arch=sm_86 flash_attention.cu -o flash_attention
```

## Usage

```bash
# Default (seq_len=1024, causal=true)
./flash_attention

# Custom sequence length
./flash_attention 4096

# Custom sequence length and non-causal
./flash_attention 2048 0
```

## The Performance Optimization Journey

This implementation went through an extensive optimization journey, starting at ~1.75 TFLOPS and eventually beating PyTorch's 21.88 TFLOPS. Here's how we got there:

### Phase 1: Foundation (V1-V10) - Getting Tensor Cores Working

**Starting Point: 1.75 TFLOPS**

The initial implementation focused on correctness using the Flash Attention algorithm with tensor cores. Key challenges included:

- Understanding the `mma.sync.aligned.m16n8k16` PTX instruction layout
- Proper fragment register mapping for FP16 inputs and FP32 outputs
- Basic tiled K/V streaming with Q kept in registers

### Phase 2: Tile Size Tuning (V11-V18) - 10x Performance Gain

**Improvement: 1.75 -> 17.5 TFLOPS**

The breakthrough came from finding the right tile sizes:

| Version | BLOCK_M | BLOCK_N | TFLOPS | Key Change |
|---------|---------|---------|--------|------------|
| V10     | 64      | 64      | 1.75   | Baseline   |
| V16     | 64      | 32      | 13.88  | Reduced BLOCK_N for better occupancy |
| V18     | 64      | 32      | 17.5   | Optimized thread/warp balance |

**Key Insight**: Smaller BLOCK_N (32 instead of 64) improved SM occupancy significantly. The K/V tiles being smaller meant less shared memory pressure and more blocks running concurrently.

### Phase 3: Micro-Optimizations (V19-V26) - Reaching 84%

**Improvement: 17.5 -> 18.4 TFLOPS (84% of PyTorch)**

Several micro-optimizations added incremental gains:

1. **Shared Memory Padding (+8)**: Added 8 elements padding to K/V shared memory strides to avoid bank conflicts
   ```cpp
   constexpr int K_STRIDE = HEAD_DIM + 8;  // 136 instead of 128
   ```

2. **Warp-Level Shuffle Reductions**: Replaced shared memory reductions with `__shfl_xor_sync` for max and sum operations
   ```cpp
   m_ij = fmaxf(m_ij, __shfl_xor_sync(0xffffffff, m_ij, 1));
   m_ij = fmaxf(m_ij, __shfl_xor_sync(0xffffffff, m_ij, 2));
   ```

3. **Simplified Causal Masking**: More efficient per-element causal checks avoiding divergent branches

4. **Proper Bounds Checking**: Clean handling of arbitrary sequence lengths with zero-padding

### Phase 4: Fast Exp Optimization - Breaking 20 TFLOPS

**Improvement: 18.4 -> 20.9 TFLOPS (95% of PyTorch)**

A critical optimization was replacing `expf()` with `exp2f()`:

```cpp
// Before (slow)
float p = expf(s - max_val);

// After (fast) - exp(x) = exp2(x * log2(e))
constexpr float LOG2E = 1.4426950408889634f;
float p = exp2f((s - max_val) * LOG2E);
```

**Why it's faster**: The CUDA special function unit (SFU) executes `exp2f` in fewer cycles than `expf`. Since attention computes millions of exponentials, this adds up significantly.

### Phase 5: Adaptive Tile Sizes - Beating PyTorch

**Final: 22.6 TFLOPS at seq_len=4096 (103% of PyTorch!)**

The final optimization was discovering that **different sequence lengths need different tile sizes**:

#### The Discovery

Benchmarking revealed the 64x32 large tile kernel underperformed at short sequences:

| seq_len | Large Tile (64x32) | Small Tile (32x32) | Winner |
|---------|-------------------|-------------------|--------|
| 256     | 7.1 TFLOPS        | 7.8 TFLOPS        | Small (+9.4%) |
| 512     | 12.7 TFLOPS       | 13.9 TFLOPS       | Small (+9.4%) |
| 768     | 16.0 TFLOPS       | 15.9 TFLOPS       | Large (+0.5%) |
| 1024    | 19.4 TFLOPS       | 19.1 TFLOPS       | Large (+1.8%) |
| 2048    | 20.9 TFLOPS       | 17.7 TFLOPS       | Large (+18%) |

**Why Small Tiles Win at Short Sequences**:
- At seq_len=512, the large tile kernel launches only 8 blocks (512/64)
- The RTX 3080 Ti has 58 SMs, so most are idle
- The small tile kernel launches 16 blocks (512/32), better utilizing the GPU
- The occupancy boost outweighs the reduced tensor core efficiency

**The Dispatcher Solution**:

```cpp
constexpr int SEQ_LEN_THRESHOLD = 768;

void flash_attention_forward(...) {
    if (seq_len < SEQ_LEN_THRESHOLD) {
        // Small tile: BLOCK_M=32, 64 threads, higher occupancy
        flash_attention_kernel_small<128, IS_CAUSAL><<<grid, 64, smem>>>(...)
    } else {
        // Large tile: BLOCK_M=64, 128 threads, better tensor core efficiency
        flash_attention_kernel_large<128, IS_CAUSAL><<<grid, 128, smem>>>(...)
    }
}
```

### Optimization Attempts That Didn't Work

Not every optimization succeeded. Here's what we tried and abandoned:

1. **Double-Buffer Prefetch**: Loading next K/V tile while computing current one
   - Result: 12 TFLOPS (vs 18 baseline) - **33% slower**
   - Reason: Kernel is compute-bound, not memory-bound. Extra sync overhead hurt.

2. **Epilogue Fusion with half2 Stores**: Vectorized 2-element writes
   - Result: 14-15 TFLOPS (vs 18 baseline) - **20% slower**
   - Reason: Register pressure from packing half2 values hurt occupancy.

3. **cp.async for Asynchronous Loads**: Modern async memory copies
   - Result: Illegal memory access errors
   - Reason: Zero-fill paths for bounds checking incompatible with async copies.

### Summary: From 1.75 to 22.6 TFLOPS

| Phase | TFLOPS | % of PyTorch | Key Optimization |
|-------|--------|--------------|------------------|
| V10   | 1.75   | 8%           | Baseline tensor cores |
| V16   | 13.88  | 63%          | BLOCK_N=32 |
| V18   | 17.5   | 80%          | Thread balance |
| V26   | 18.4   | 84%          | Micro-optimizations |
| +exp2 | 20.9   | 95%          | Fast exp2f |
| +tiles| 22.6   | **103%**     | Adaptive dispatcher |

**Total speedup: 12.9x from initial to final**

## Algorithm

Implements the Flash Attention algorithm from ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135):

1. Tile Q, K, V matrices to fit in shared memory
2. Keep Q tile in registers, stream K/V tiles through shared memory
3. Compute QK^T using tensor cores (`mma.sync.m16n8k16`)
4. Apply online softmax (tracking running max and sum via warp shuffles)
5. Accumulate P @ V using tensor cores
6. Rescale output by final softmax denominator

## File Structure

```
cuda/
  flash_attention.cu       # Main implementation with dispatcher
  flash_attention_kernel.cuh  # Kernel header (for testing)
  test_kernel.cu           # Experimental testing harness
  bench_graph.cu           # CUDA graph benchmarking
  Makefile                 # Build configuration
```

## License

MIT
