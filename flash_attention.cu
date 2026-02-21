/**
 * Flash Attention V9 - Adaptive V loading + Split-K
 *
 * Two V-loading strategies selected at dispatch time:
 * - V_LDMATRIX=false: V8-style scalar loads from padded smem (best for short seq)
 * - V_LDMATRIX=true:  ldmatrix.x2.trans from XOR-swizzled smem (best for long seq)
 *
 * K always uses padded stride with scalar loads (already zero bank conflicts).
 *
 * Target: RTX 3080 Ti Laptop (SM86, 58 SMs)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <unistd.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__device__ __forceinline__ uint32_t smem_ptr(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(addr) : "l"(ptr));
    return addr;
}

// cp.async helpers for SM80+ (Ampere)
__device__ __forceinline__ void cp_async_16B(uint32_t dst_smem, const void* src_global, int src_size) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst_smem), "l"(src_global), "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group 0;\n");
}
__device__ __forceinline__ void cp_async_wait_one() {
    asm volatile("cp.async.wait_group 1;\n");
}
__device__ __forceinline__ void cp_async_wait_two() {
    asm volatile("cp.async.wait_group 2;\n");
}

// ============================================================================
// V9 kernel - V_LDMATRIX selects V loading strategy
// ============================================================================
// GRID_SWAP: false = grid(nb, bh), true = grid(bh, nb) with reversed Q-blocks for causal
// K_SWIZZLE: true = K uses XOR swizzle (K_STRIDE=HD), false = K uses +8 padding (K_STRIDE=HD+8)
// K_LDMATRIX: true = use ldmatrix.x2.trans for K loads in QK matmul
// ASYNC_PIPELINE: true = use cp.async to overlap V load with QK, K[next] load with PV
template<int BLOCK_M, int BLOCK_N, int NWARPS, int HEAD_DIM,
         bool IS_CAUSAL, int MIN_BLOCKS, bool IS_SPLITK, bool V_LDMATRIX,
         bool GRID_SWAP = false, bool K_SWIZZLE = false, bool K_LDMATRIX = false,
         bool ASYNC_PIPELINE = false>
__global__ void __launch_bounds__(NWARPS * 32, MIN_BLOCKS) flash_attention_v9(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ Output,
    float* __restrict__ O_partial,
    float* __restrict__ ml_partial,
    const int seq_len,
    const float scale,
    const int split_k
) {
    constexpr int BLOCK_SIZE = NWARPS * 32;
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int N_TILES = BLOCK_N / MMA_N;
    constexpr int K_TILES = HEAD_DIM / MMA_K;
    constexpr int O_TILES = HEAD_DIM / MMA_N;
    constexpr int PV_K_TILES = BLOCK_N / MMA_K;

    constexpr int K_STRIDE = K_SWIZZLE ? HEAD_DIM : (HEAD_DIM + 8);
    // V stride: padded for scalar (eliminates 4-way bank conflicts in PV),
    // unpadded for ldmatrix (XOR swizzle handles conflicts)
    constexpr int V_STRIDE = V_LDMATRIX ? HEAD_DIM : (HEAD_DIM + 8);
    constexpr int V_ROW_BYTES = V_STRIDE * 2;


    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int groupID = lane_id / 4;
    const int threadID = lane_id % 4;

    int batch_head, block_m;
    if constexpr (GRID_SWAP) {
        batch_head = blockIdx.x;
        block_m = IS_CAUSAL ? (gridDim.y - 1 - blockIdx.y) * BLOCK_M
                            : blockIdx.y * BLOCK_M;
    } else {
        block_m = blockIdx.x * BLOCK_M;
        batch_head = blockIdx.y;
    }
    if (block_m >= seq_len) return;

    const int warp_row_start = warp_id * MMA_M;
    const int global_row_base = block_m + warp_row_start;
    const int global_row_a = global_row_base + groupID;
    const int global_row_b = global_row_base + groupID + 8;

    const int stride = seq_len * HEAD_DIM;
    const half* Q_base = Q + batch_head * stride + global_row_base * HEAD_DIM;
    const half* K_base = K + batch_head * stride;
    const half* V_base = V + batch_head * stride;

    extern __shared__ char smem[];
    half* smem_base = (half*)smem;

    half* K_smem = smem_base;
    half* V_smem = K_smem + BLOCK_N * K_STRIDE;

    uint32_t V_smem_addr = smem_ptr(V_smem);
    uint32_t K_smem_addr = smem_ptr(K_smem);
    constexpr int K_ROW_BYTES = K_STRIDE * 2;

    // ldmatrix constants for K and V
    int ld_lane, ld_lane_row8;
    if constexpr (V_LDMATRIX || K_LDMATRIX) {
        ld_lane = min(lane_id, 15);
        ld_lane_row8 = ld_lane & 7;
    }

    float O_frag[O_TILES][4] = {{0.0f}};
    float m_i[2] = {-FLT_MAX, -FLT_MAX};
    float l_i[2] = {0.0f, 0.0f};

    // Load Q fragments to registers (all k-tiles pre-loaded)
    uint32_t Q_frag[K_TILES][4];
    #pragma unroll
    for (int k = 0; k < K_TILES; k++) {
        int q_col_lo = k * MMA_K + threadID * 2;
        int q_col_hi = k * MMA_K + threadID * 2 + 8;
        if (global_row_a < seq_len) {
            Q_frag[k][0] = *reinterpret_cast<const uint32_t*>(Q_base + groupID * HEAD_DIM + q_col_lo);
            Q_frag[k][1] = *reinterpret_cast<const uint32_t*>(Q_base + groupID * HEAD_DIM + q_col_hi);
        } else { Q_frag[k][0] = 0; Q_frag[k][1] = 0; }
        if (global_row_b < seq_len) {
            Q_frag[k][2] = *reinterpret_cast<const uint32_t*>(Q_base + (groupID+8) * HEAD_DIM + q_col_lo);
            Q_frag[k][3] = *reinterpret_cast<const uint32_t*>(Q_base + (groupID+8) * HEAD_DIM + q_col_hi);
        } else { Q_frag[k][2] = 0; Q_frag[k][3] = 0; }
    }

    // KV loop bounds
    int total_kv_end;
    if constexpr (IS_CAUSAL) {
        total_kv_end = min(block_m + BLOCK_M, seq_len);
    } else {
        total_kv_end = seq_len;
    }

    int kv_begin, kv_end;
    if constexpr (IS_SPLITK) {
        int total_kv_blocks = (total_kv_end + BLOCK_N - 1) / BLOCK_N;
        int blocks_per_split = (total_kv_blocks + split_k - 1) / split_k;
        int split_idx = blockIdx.z;
        kv_begin = split_idx * blocks_per_split * BLOCK_N;
        kv_end = min((split_idx + 1) * blocks_per_split * BLOCK_N, total_kv_end);
        if (kv_begin >= total_kv_end) return;
    } else {
        kv_begin = 0;
        kv_end = total_kv_end;
    }

    // ======== Main KV loop ========
    constexpr int THREADS_PER_ROW = HEAD_DIM / 8;  // 16
    constexpr int ROWS_PER_COPY = BLOCK_SIZE / THREADS_PER_ROW;
    const int copy_col = (threadIdx.x % THREADS_PER_ROW) * 8;

    // --- Macros for QK, Softmax, PV (shared between sync and async paths) ---
    #define DO_QK_MATMUL() \
        float S_frag[N_TILES][4]; \
        for (int n = 0; n < N_TILES; n++) { \
            S_frag[n][0]=0; S_frag[n][1]=0; S_frag[n][2]=0; S_frag[n][3]=0; \
        } \
        for (int k = 0; k < K_TILES; k++) { \
            uint32_t Q_k[4]; \
            Q_k[0] = Q_frag[k][0]; Q_k[1] = Q_frag[k][1]; \
            Q_k[2] = Q_frag[k][2]; Q_k[3] = Q_frag[k][3]; \
            int k_col_lo, k_col_hi; \
            if constexpr (!K_LDMATRIX) { \
                k_col_lo = k * MMA_K + threadID * 2; \
                k_col_hi = k * MMA_K + threadID * 2 + 8; \
            } \
            for (int n = 0; n < N_TILES; n++) { \
                uint32_t K_mma[2]; \
                if constexpr (K_LDMATRIX) { \
                    int k_kv_row = n * MMA_N + ld_lane_row8; \
                    int k_hdim_group = k * 2 + (ld_lane >> 3); \
                    uint32_t k_addr = K_smem_addr + k_kv_row * K_ROW_BYTES + k_hdim_group * 16; \
                    asm volatile( \
                        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n" \
                        : "=r"(K_mma[0]), "=r"(K_mma[1]) : "r"(k_addr)); \
                } else if constexpr (K_SWIZZLE) { \
                    int k_row = n * MMA_N + groupID; \
                    int k_row7 = k_row & 7; \
                    int sw_col_lo = ((k_col_lo >> 3) ^ k_row7) * 8 + (k_col_lo & 7); \
                    int sw_col_hi = ((k_col_hi >> 3) ^ k_row7) * 8 + (k_col_hi & 7); \
                    K_mma[0] = *reinterpret_cast<uint32_t*>(&K_smem[k_row * K_STRIDE + sw_col_lo]); \
                    K_mma[1] = *reinterpret_cast<uint32_t*>(&K_smem[k_row * K_STRIDE + sw_col_hi]); \
                } else { \
                    int k_row = n * MMA_N + groupID; \
                    K_mma[0] = *reinterpret_cast<uint32_t*>(&K_smem[k_row * K_STRIDE + k_col_lo]); \
                    K_mma[1] = *reinterpret_cast<uint32_t*>(&K_smem[k_row * K_STRIDE + k_col_hi]); \
                } \
                asm volatile( \
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n" \
                    : "=f"(S_frag[n][0]),"=f"(S_frag[n][1]), \
                      "=f"(S_frag[n][2]),"=f"(S_frag[n][3]) \
                    : "r"(Q_k[0]),"r"(Q_k[1]),"r"(Q_k[2]),"r"(Q_k[3]), \
                      "r"(K_mma[0]),"r"(K_mma[1]), \
                      "f"(S_frag[n][0]),"f"(S_frag[n][1]), \
                      "f"(S_frag[n][2]),"f"(S_frag[n][3])); \
            } \
        }

    #define DO_SOFTMAX(kv_start_val) \
        { \
        float m_ij_a = -FLT_MAX, m_ij_b = -FLT_MAX; \
        constexpr float NEG_INF = -10000.0f; \
        constexpr float LOG2E = 1.4426950408889634f; \
        for (int n = 0; n < N_TILES; n++) { \
            int base_col = (kv_start_val) + n * MMA_N + threadID * 2; \
            S_frag[n][0]*=scale; S_frag[n][1]*=scale; \
            S_frag[n][2]*=scale; S_frag[n][3]*=scale; \
            if (base_col >= seq_len) { \
                S_frag[n][0]=NEG_INF; S_frag[n][1]=NEG_INF; \
                S_frag[n][2]=NEG_INF; S_frag[n][3]=NEG_INF; \
            } else if (base_col+1 >= seq_len) { \
                S_frag[n][1]=NEG_INF; S_frag[n][3]=NEG_INF; \
            } \
            if constexpr (IS_CAUSAL) { \
                S_frag[n][0]=(base_col<=global_row_a)?S_frag[n][0]:NEG_INF; \
                S_frag[n][1]=(base_col+1<=global_row_a)?S_frag[n][1]:NEG_INF; \
                S_frag[n][2]=(base_col<=global_row_b)?S_frag[n][2]:NEG_INF; \
                S_frag[n][3]=(base_col+1<=global_row_b)?S_frag[n][3]:NEG_INF; \
            } \
            m_ij_a = fmaxf(m_ij_a, fmaxf(S_frag[n][0], S_frag[n][1])); \
            m_ij_b = fmaxf(m_ij_b, fmaxf(S_frag[n][2], S_frag[n][3])); \
        } \
        m_ij_a = fmaxf(m_ij_a, __shfl_xor_sync(0xffffffff, m_ij_a, 1)); \
        m_ij_a = fmaxf(m_ij_a, __shfl_xor_sync(0xffffffff, m_ij_a, 2)); \
        m_ij_b = fmaxf(m_ij_b, __shfl_xor_sync(0xffffffff, m_ij_b, 1)); \
        m_ij_b = fmaxf(m_ij_b, __shfl_xor_sync(0xffffffff, m_ij_b, 2)); \
        float m_new_a = fmaxf(m_i[0], m_ij_a); \
        float m_new_b = fmaxf(m_i[1], m_ij_b); \
        float alpha_a = (m_i[0] > -FLT_MAX/2) ? exp2f((m_i[0]-m_new_a)*LOG2E) : 0.0f; \
        float alpha_b = (m_i[1] > -FLT_MAX/2) ? exp2f((m_i[1]-m_new_b)*LOG2E) : 0.0f; \
        for (int o = 0; o < O_TILES; o++) { \
            O_frag[o][0]*=alpha_a; O_frag[o][1]*=alpha_a; \
            O_frag[o][2]*=alpha_b; O_frag[o][3]*=alpha_b; \
        } \
        float l_ij_a=0, l_ij_b=0; \
        for (int n = 0; n < N_TILES; n++) { \
            S_frag[n][0]=exp2f((S_frag[n][0]-m_new_a)*LOG2E); \
            S_frag[n][1]=exp2f((S_frag[n][1]-m_new_a)*LOG2E); \
            S_frag[n][2]=exp2f((S_frag[n][2]-m_new_b)*LOG2E); \
            S_frag[n][3]=exp2f((S_frag[n][3]-m_new_b)*LOG2E); \
            l_ij_a += S_frag[n][0]+S_frag[n][1]; \
            l_ij_b += S_frag[n][2]+S_frag[n][3]; \
        } \
        l_ij_a += __shfl_xor_sync(0xffffffff, l_ij_a, 1); \
        l_ij_a += __shfl_xor_sync(0xffffffff, l_ij_a, 2); \
        l_ij_b += __shfl_xor_sync(0xffffffff, l_ij_b, 1); \
        l_ij_b += __shfl_xor_sync(0xffffffff, l_ij_b, 2); \
        l_i[0] = l_i[0]*alpha_a + l_ij_a; \
        l_i[1] = l_i[1]*alpha_b + l_ij_b; \
        m_i[0] = m_new_a; \
        m_i[1] = m_new_b; \
        }

    #define DO_PV_MATMUL() \
        for (int o = 0; o < O_TILES; o++) { \
            for (int t = 0; t < PV_K_TILES; t++) { \
                half2 p0, p1, p2, p3; \
                p0 = make_half2(__float2half(S_frag[2*t][0]), __float2half(S_frag[2*t][1])); \
                p1 = make_half2(__float2half(S_frag[2*t+1][0]), __float2half(S_frag[2*t+1][1])); \
                p2 = make_half2(__float2half(S_frag[2*t][2]), __float2half(S_frag[2*t][3])); \
                p3 = make_half2(__float2half(S_frag[2*t+1][2]), __float2half(S_frag[2*t+1][3])); \
                uint32_t P_mma[4]; \
                P_mma[0] = *reinterpret_cast<uint32_t*>(&p0); \
                P_mma[1] = *reinterpret_cast<uint32_t*>(&p2); \
                P_mma[2] = *reinterpret_cast<uint32_t*>(&p1); \
                P_mma[3] = *reinterpret_cast<uint32_t*>(&p3); \
                uint32_t V_mma[2]; \
                if constexpr (V_LDMATRIX) { \
                    int v_row = t * MMA_K + ld_lane; \
                    int v_sw = o ^ ld_lane_row8; \
                    uint32_t v_addr = V_smem_addr + v_row * V_ROW_BYTES + v_sw * 16; \
                    asm volatile( \
                        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n" \
                        : "=r"(V_mma[0]), "=r"(V_mma[1]) : "r"(v_addr)); \
                } else { \
                    int v_head_col = o * MMA_N + groupID; \
                    int v_key_base = t * MMA_K; \
                    int v_k0 = threadID * 2; \
                    half vh0 = V_smem[(v_key_base+v_k0)  *V_STRIDE+v_head_col]; \
                    half vh1 = V_smem[(v_key_base+v_k0+1)*V_STRIDE+v_head_col]; \
                    half vh2 = V_smem[(v_key_base+v_k0+8)*V_STRIDE+v_head_col]; \
                    half vh3 = V_smem[(v_key_base+v_k0+9)*V_STRIDE+v_head_col]; \
                    half2 v_pair0 = make_half2(vh0, vh1); \
                    half2 v_pair1 = make_half2(vh2, vh3); \
                    V_mma[0] = *reinterpret_cast<uint32_t*>(&v_pair0); \
                    V_mma[1] = *reinterpret_cast<uint32_t*>(&v_pair1); \
                } \
                asm volatile( \
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n" \
                    : "=f"(O_frag[o][0]),"=f"(O_frag[o][1]), \
                      "=f"(O_frag[o][2]),"=f"(O_frag[o][3]) \
                    : "r"(P_mma[0]),"r"(P_mma[1]),"r"(P_mma[2]),"r"(P_mma[3]), \
                      "r"(V_mma[0]),"r"(V_mma[1]), \
                      "f"(O_frag[o][0]),"f"(O_frag[o][1]), \
                      "f"(O_frag[o][2]),"f"(O_frag[o][3])); \
            } \
        }

    if constexpr (ASYNC_PIPELINE) {
        // ======== 3-Stage cp.async pipeline ========
        // 3 K+V buffers. cp.async.wait_group 2 keeps 2 loads in flight.
        // Same L1 as 2-stage (both use 100KB smem partition on SM86).
        constexpr int BUF_HALVES = BLOCK_N * K_STRIDE + BLOCK_N * V_STRIDE;
        constexpr int BUF_BYTES = BUF_HALVES * 2;
        constexpr int CHUNKS_PER_ROW = HEAD_DIM / 8;
        constexpr int TOTAL_CHUNKS = BLOCK_N * CHUNKS_PER_ROW;
        const uint32_t smem_base_addr = smem_ptr(smem_base);

        #define ASYNC_LOAD_TILE(kv_off, b) { \
            const uint32_t _k_addr = smem_base_addr + (b) * BUF_BYTES; \
            const uint32_t _v_addr = _k_addr + BLOCK_N * K_ROW_BYTES; \
            _Pragma("unroll 4") \
            for (int _i = threadIdx.x; _i < TOTAL_CHUNKS; _i += BLOCK_SIZE) { \
                int _row = _i / CHUNKS_PER_ROW; \
                int _chunk = _i % CHUNKS_PER_ROW; \
                int _gr = (kv_off) + _row; \
                int _sz = (_gr < seq_len) ? 16 : 0; \
                cp_async_16B(_k_addr + _row * K_ROW_BYTES + _chunk * 16, \
                             (const void*)&K_base[_gr * HEAD_DIM + _chunk * 8], _sz); \
            } \
            _Pragma("unroll 4") \
            for (int _i = threadIdx.x; _i < TOTAL_CHUNKS; _i += BLOCK_SIZE) { \
                int _row = _i / CHUNKS_PER_ROW; \
                int _chunk = _i % CHUNKS_PER_ROW; \
                int _gr = (kv_off) + _row; \
                int _sz = (_gr < seq_len) ? 16 : 0; \
                int _sw = V_LDMATRIX ? (_chunk ^ (_row & 7)) : _chunk; \
                cp_async_16B(_v_addr + _row * V_ROW_BYTES + _sw * 16, \
                             (const void*)&V_base[_gr * HEAD_DIM + _chunk * 8], _sz); \
            } \
        }

        // Prologue: fill up to 3 buffers
        int num_tiles = (kv_end - kv_begin + BLOCK_N - 1) / BLOCK_N;
        ASYNC_LOAD_TILE(kv_begin, 0);
        cp_async_commit();
        if (num_tiles >= 2) { ASYNC_LOAD_TILE(kv_begin + BLOCK_N, 1); cp_async_commit(); }
        if (num_tiles >= 3) { ASYNC_LOAD_TILE(kv_begin + 2*BLOCK_N, 2); cp_async_commit(); }

        // Wait for buf[0], keep others in flight
        if (num_tiles >= 3)      cp_async_wait_two();
        else if (num_tiles >= 2) cp_async_wait_one();
        else                     cp_async_wait();
        __syncthreads();

        int buf = 0;
        for (int kv_start = kv_begin; kv_start < kv_end; kv_start += BLOCK_N) {
            // Point to current buffer
            K_smem = smem_base + buf * BUF_HALVES;
            V_smem = K_smem + BLOCK_N * K_STRIDE;
            K_smem_addr = smem_base_addr + buf * BUF_BYTES;
            V_smem_addr = K_smem_addr + BLOCK_N * K_ROW_BYTES;

            // Compute on current buffer
            DO_QK_MATMUL();
            DO_SOFTMAX(kv_start);
            DO_PV_MATMUL();

            // Prefetch tile i+3 into current buffer (safe: last read 2 iters ago)
            int prefetch = kv_start + 3 * BLOCK_N;
            bool did_prefetch = (prefetch < kv_end);
            if (did_prefetch) {
                ASYNC_LOAD_TILE(prefetch, buf);
                cp_async_commit();
            }

            // Advance buffer: 0->1->2->0
            buf = (buf == 2) ? 0 : buf + 1;

            // Wait for next buffer
            if (kv_start + BLOCK_N < kv_end) {
                if (did_prefetch) cp_async_wait_two();  // 3 in flight -> keep 2
                else              cp_async_wait();       // drain all remaining
                __syncthreads();
            }
        }
        #undef ASYNC_LOAD_TILE
    } else {
        // ======== Original synchronous KV loop ========
        for (int kv_start = kv_begin; kv_start < kv_end; kv_start += BLOCK_N) {

            // --- Load K and V to shared memory ---
            for (int row = threadIdx.x / THREADS_PER_ROW; row < BLOCK_N; row += ROWS_PER_COPY) {
                int global_row = kv_start + row;

                float4 k_data;
                if (global_row < seq_len)
                    k_data = *reinterpret_cast<const float4*>(&K_base[global_row * HEAD_DIM + copy_col]);
                else
                    memset(&k_data, 0, sizeof(float4));
                if constexpr (K_SWIZZLE) {
                    int k_sw_col = ((copy_col >> 3) ^ (row & 7)) << 3;
                    *reinterpret_cast<float4*>(&K_smem[row * K_STRIDE + k_sw_col]) = k_data;
                } else {
                    *reinterpret_cast<float4*>(&K_smem[row * K_STRIDE + copy_col]) = k_data;
                }

                float4 v_data;
                if (global_row < seq_len)
                    v_data = *reinterpret_cast<const float4*>(&V_base[global_row * HEAD_DIM + copy_col]);
                else
                    memset(&v_data, 0, sizeof(float4));
                if constexpr (V_LDMATRIX) {
                    int v_sw_col = ((copy_col >> 3) ^ (row & 7)) << 3;
                    *reinterpret_cast<float4*>(&V_smem[row * V_STRIDE + v_sw_col]) = v_data;
                } else {
                    *reinterpret_cast<float4*>(&V_smem[row * V_STRIDE + copy_col]) = v_data;
                }
            }
            __syncthreads();

            DO_QK_MATMUL();
            DO_SOFTMAX(kv_start);
            DO_PV_MATMUL();
            __syncthreads();
        }
    }
    #undef DO_QK_MATMUL
    #undef DO_SOFTMAX
    #undef DO_PV_MATMUL

    // ======== Output ========
    if constexpr (IS_SPLITK) {
        int split_idx = blockIdx.z;
        int split_stride;
        if constexpr (GRID_SWAP) {
            split_stride = gridDim.x * seq_len;  // gridDim.x = batch_heads
        } else {
            split_stride = gridDim.y * seq_len;  // gridDim.y = batch_heads
        }
        int row_a = batch_head * seq_len + global_row_base + groupID;
        int row_b = batch_head * seq_len + global_row_base + groupID + 8;
        int o_base_a = (split_idx * split_stride + row_a) * HEAD_DIM;
        int o_base_b = (split_idx * split_stride + row_b) * HEAD_DIM;
        int ml_base_a = (split_idx * split_stride + row_a) * 2;
        int ml_base_b = (split_idx * split_stride + row_b) * 2;

        #pragma unroll
        for (int o = 0; o < O_TILES; o++) {
            int col = o * MMA_N + threadID * 2;
            if (global_row_base + groupID < seq_len) {
                O_partial[o_base_a + col]     = O_frag[o][0];
                O_partial[o_base_a + col + 1] = O_frag[o][1];
            }
            if (global_row_base + groupID + 8 < seq_len) {
                O_partial[o_base_b + col]     = O_frag[o][2];
                O_partial[o_base_b + col + 1] = O_frag[o][3];
            }
        }
        if (threadID == 0) {
            if (global_row_base + groupID < seq_len) {
                ml_partial[ml_base_a]     = m_i[0];
                ml_partial[ml_base_a + 1] = l_i[0];
            }
            if (global_row_base + groupID + 8 < seq_len) {
                ml_partial[ml_base_b]     = m_i[1];
                ml_partial[ml_base_b + 1] = l_i[1];
            }
        }
    } else {
        // Stage output through shared memory for coalesced float4 global writes.
        // Multi-pass when smem < BLOCK_M × HEAD_DIM (e.g., BN=32 with BM=128).
        // BN=64: 1 pass (all warps at once). BN=32: 2 passes (4 warps each).
        half* O_smem = (half*)smem;
        float inv_l_a = (l_i[0] > 0) ? 1.0f / l_i[0] : 0.0f;
        float inv_l_b = (l_i[1] > 0) ? 1.0f / l_i[1] : 0.0f;

        constexpr int KV_SMEM_HALVES = BLOCK_N * K_STRIDE + BLOCK_N * V_STRIDE;
        constexpr int MAX_WARPS_FIT = KV_SMEM_HALVES / (MMA_M * HEAD_DIM);
        constexpr int WPP = MAX_WARPS_FIT > NWARPS ? NWARPS : MAX_WARPS_FIT;
        constexpr int ROWS_PER_PASS = WPP * MMA_M;
        constexpr int O_PASSES = (BLOCK_M + ROWS_PER_PASS - 1) / ROWS_PER_PASS;

        half* O_global = Output + batch_head * stride + block_m * HEAD_DIM;
        constexpr int F4_PER_ROW = HEAD_DIM / 8;
        constexpr int ROWS_PER_ITER = BLOCK_SIZE / F4_PER_ROW;

        #pragma unroll
        for (int pass = 0; pass < O_PASSES; pass++) {
            const int pass_warp_start = pass * WPP;
            const int pass_row_offset = pass * ROWS_PER_PASS;

            // Only warps belonging to this pass scatter their O_frag to smem
            if (warp_id >= pass_warp_start && warp_id < pass_warp_start + WPP) {
                const int local_row = (warp_id - pass_warp_start) * MMA_M;
                #pragma unroll
                for (int o = 0; o < O_TILES; o++) {
                    int col = o * MMA_N + threadID * 2;
                    half2 va = make_half2(__float2half(O_frag[o][0] * inv_l_a),
                                           __float2half(O_frag[o][1] * inv_l_a));
                    half2 vb = make_half2(__float2half(O_frag[o][2] * inv_l_b),
                                           __float2half(O_frag[o][3] * inv_l_b));
                    *reinterpret_cast<uint32_t*>(&O_smem[(local_row + groupID) * HEAD_DIM + col]) =
                        *reinterpret_cast<uint32_t*>(&va);
                    *reinterpret_cast<uint32_t*>(&O_smem[(local_row + groupID + 8) * HEAD_DIM + col]) =
                        *reinterpret_cast<uint32_t*>(&vb);
                }
            }
            __syncthreads();

            // All threads cooperatively write ROWS_PER_PASS rows with coalesced float4
            #pragma unroll
            for (int i = 0; i < ROWS_PER_PASS / ROWS_PER_ITER; i++) {
                int row = i * ROWS_PER_ITER + threadIdx.x / F4_PER_ROW;
                int col = (threadIdx.x % F4_PER_ROW) * 8;
                if (block_m + pass_row_offset + row < seq_len) {
                    *reinterpret_cast<float4*>(&O_global[(pass_row_offset + row) * HEAD_DIM + col]) =
                        *reinterpret_cast<float4*>(&O_smem[row * HEAD_DIM + col]);
                }
            }

            if constexpr (O_PASSES > 1) {
                if (pass < O_PASSES - 1) __syncthreads();
            }
        }
    }
}

// ============================================================================
// Split-K merge kernel
// ============================================================================
template<int HEAD_DIM, int MAX_SPLITS>
__global__ void flash_attention_splitk_merge(
    const float* __restrict__ O_partial,
    const float* __restrict__ ml_partial,
    half* __restrict__ Output,
    const int seq_len,
    const int batch_heads,
    const int split_k
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int d_base = threadIdx.x * 4;
    if (row >= batch_heads * seq_len || d_base >= HEAD_DIM) return;

    const int split_stride = batch_heads * seq_len;
    constexpr float LOG2E = 1.4426950408889634f;

    float m_vals[MAX_SPLITS], l_vals[MAX_SPLITS];
    float m_max = -FLT_MAX;
    for (int s = 0; s < split_k; s++) {
        int ml_idx = (s * split_stride + row) * 2;
        m_vals[s] = ml_partial[ml_idx];
        l_vals[s] = ml_partial[ml_idx + 1];
        m_max = fmaxf(m_max, m_vals[s]);
    }

    float o_sum[4] = {0,0,0,0};
    float l_sum = 0;
    for (int s = 0; s < split_k; s++) {
        float w = (m_vals[s] > -FLT_MAX/2) ? exp2f((m_vals[s] - m_max) * LOG2E) : 0.0f;
        l_sum += l_vals[s] * w;
        int o_idx = (s * split_stride + row) * HEAD_DIM + d_base;
        for (int i = 0; i < 4; i++)
            o_sum[i] += O_partial[o_idx + i] * w;
    }

    float inv_l = (l_sum > 0) ? 1.0f / l_sum : 0.0f;
    int out_idx = row * HEAD_DIM + d_base;
    for (int i = 0; i < 4; i++)
        Output[out_idx + i] = __float2half(o_sum[i] * inv_l);
}

// ============================================================================
// Dispatch - Adaptive BM + Adaptive V loading
//   seq < 1024:  BM=64,  4 warps, 2 blocks/SM (better occupancy, latency hiding)
//   seq >= 1024: BM=128, 8 warps, 1 block/SM (40-49% less memory traffic)
//   V loading:   scalar for seq < 512, ldmatrix for seq >= 512
// ============================================================================
void flash_attention_v9_dispatch(
    const half* Q, const half* K, const half* V, half* Output,
    float* splitk_buf_O, float* splitk_buf_ml,
    int batch_size, int num_heads, int seq_len, int head_dim,
    bool causal, cudaStream_t stream = 0
) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    constexpr int BN = 64, HD = 128;

    int batch_heads = batch_size * num_heads;
    // All paths: K_LDMATRIX + V_LDMATRIX, K padded (K_STRIDE=HD+8)
    constexpr int K_STRIDE_C = HD + 8;  // 136
    size_t smem = (BN * K_STRIDE_C + BN * HD) * sizeof(half);  // 33,792 bytes

    // Four-tier dispatch:
    // - Causal short (<2048):      BM=64,  BN=64,  grid(bh,nb)+reversed, ldKV, 2blk/SM
    // - Causal long (>=2048):      BM=128, BN=64,  grid(nb,bh), ldV, 1blk/SM
    // - Non-causal short (<2048):  BM=64,  BN=64,  grid(nb,bh), ldKV, 2blk/SM
    // - Non-causal long (>=2048):  BM=128, BN=128, grid(nb,bh), ldV, 1blk/SM

    if (!causal && seq_len >= 2048) {
        // Non-causal BM=128/BN=128: fewer KV tiles, 238 regs, 67.5KB smem, 1blk/SM
        constexpr int BM = 128, BN_L = 128, NWARPS = 8, BS = 256;
        constexpr size_t smem_l = (BN_L * K_STRIDE_C + BN_L * HD) * sizeof(half); // 67,584
        int nb = (seq_len + BM - 1) / BM;
        dim3 grid(nb, batch_heads);
        auto kernel = flash_attention_v9<BM,BN_L,NWARPS,HD,false,1,false,true,false,false,false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_l);
        kernel<<<grid, BS, smem_l, stream>>>(Q,K,V,Output,nullptr,nullptr,seq_len,scale,1);
    } else if (causal && seq_len >= 2048) {
        // Causal BM=128/BN=64
        constexpr int BM = 128, NWARPS = 8, BS = 256;
        int nb = (seq_len + BM - 1) / BM;
        dim3 grid(nb, batch_heads);
        auto kernel = flash_attention_v9<BM,BN,NWARPS,HD,true,1,false,true,false,false,false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kernel<<<grid, BS, smem, stream>>>(Q,K,V,Output,nullptr,nullptr,seq_len,scale,1);
    } else if (causal) {
        // Causal BM=64: grid(bh,nb)+reversed for work-balanced SM assignment
        constexpr int BM = 64, NWARPS = 4, BS = 128;
        int nb = (seq_len + BM - 1) / BM;
        dim3 grid(batch_heads, nb);

        auto kernel = flash_attention_v9<BM,BN,NWARPS,HD,true,2,false,true,true,false,true>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kernel<<<grid, BS, smem, stream>>>(Q,K,V,Output,nullptr,nullptr,seq_len,scale,1);
    } else {
        // Non-causal BM=64/BN=64: grid(nb,bh) for L2 locality
        constexpr int BM = 64, NWARPS = 4, BS = 128;
        int nb = (seq_len + BM - 1) / BM;
        dim3 grid(nb, batch_heads);

        auto kernel = flash_attention_v9<BM,BN,NWARPS,HD,false,2,false,true,false,false,true>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kernel<<<grid, BS, smem, stream>>>(Q,K,V,Output,nullptr,nullptr,seq_len,scale,1);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// CPU reference
// ============================================================================
void cpu_attention(const half* Q, const half* K, const half* V, half* Output,
    int batch_size, int num_heads, int seq_len, int head_dim, bool causal) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    for (int bh = 0; bh < batch_size * num_heads; bh++) {
        const half* q = Q + bh * seq_len * head_dim;
        const half* k = K + bh * seq_len * head_dim;
        const half* v = V + bh * seq_len * head_dim;
        half* o = Output + bh * seq_len * head_dim;
        float* scores = (float*)malloc(seq_len * sizeof(float));
        for (int i = 0; i < seq_len; i++) {
            float max_val = -FLT_MAX;
            int end_j = causal ? i + 1 : seq_len;
            for (int j = 0; j < end_j; j++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    score += __half2float(q[i*head_dim+d]) * __half2float(k[j*head_dim+d]);
                score *= scale; scores[j] = score; max_val = fmaxf(max_val, score);
            }
            float sum = 0.0f;
            for (int j = 0; j < end_j; j++) { scores[j] = expf(scores[j]-max_val); sum += scores[j]; }
            for (int j = 0; j < end_j; j++) scores[j] /= sum;
            for (int d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (int j = 0; j < end_j; j++) val += scores[j] * __half2float(v[j*head_dim+d]);
                o[i*head_dim+d] = __float2half(val);
            }
        }
        free(scores);
    }
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("=== Flash Attention V9 - Adaptive BM ===\n");
    printf("  Causal:     seq < 2048: BM=64/2blk, seq >= 2048: BM=128/1blk\n");
    printf("  Non-causal: seq < 2048: BM=64/2blk, >= 2048: BM=128/BN=128/1blk\n");
    printf("batch=1, head_dim=128\n\n");

    const int batch_size = 1, head_dim = 128;
    const bool causal = true;

    // Register info
    {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, flash_attention_v9<64,64,4,128,true,2,false,true,true,false,true>);
        printf("BM64  causal ldKV (swap):     %d regs, %zu spill\n", attr.numRegs, attr.localSizeBytes);
        cudaFuncGetAttributes(&attr, flash_attention_v9<128,64,8,128,true,1,false,true,false,false,false>);
        printf("BM128 causal ldV  (1blk):     %d regs, %zu spill\n", attr.numRegs, attr.localSizeBytes);
        cudaFuncGetAttributes(&attr, flash_attention_v9<128,64,8,128,false,1,false,true,false,false,false>);
        printf("BM128 noncausal BN64 (1blk):  %d regs, %zu spill\n", attr.numRegs, attr.localSizeBytes);
        cudaFuncGetAttributes(&attr, flash_attention_v9<128,128,8,128,false,1,false,true,false,false,false>);
        printf("BM128 noncausal BN128 (1blk): %d regs, %zu spill\n", attr.numRegs, attr.localSizeBytes);
        cudaFuncGetAttributes(&attr, flash_attention_v9<64,64,4,128,false,2,false,true,false,false,true>);
        printf("BM64  noncausal BN64  (2blk): %d regs, %zu spill\n", attr.numRegs, attr.localSizeBytes);
        cudaFuncGetAttributes(&attr, flash_attention_v9<128,64,8,128,false,1,false,true,false,false,false,true>);
        printf("BM128 3-stage pipeline(1blk): %d regs, %zu spill\n", attr.numRegs, attr.localSizeBytes);
        // Occupancy checks
        int numBlocks;
        constexpr size_t smem_check = (64 * 136 + 64 * 128) * 2;  // 33,792 bytes
        constexpr size_t smem_bn128 = (128 * 136 + 128 * 128) * 2; // 67,584 bytes
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
            flash_attention_v9<64,64,4,128,false,2,false,true,false,false,true>, 128, smem_check);
        printf("BM64  occupancy: %d blk/SM\n", numBlocks);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
            flash_attention_v9<128,64,8,128,false,1,false,true,false,false,false>, 256, smem_check);
        printf("BM128/BN64 occupancy: %d blk/SM\n", numBlocks);
        {
            auto bn128_k = flash_attention_v9<128,128,8,128,false,1,false,true,false,false,false>;
            cudaFuncSetAttribute(bn128_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bn128);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, bn128_k, 256, smem_bn128);
            printf("BM128/BN128 occupancy: %d blk/SM\n", numBlocks);
        }
        {
            auto bn64_nc_k = flash_attention_v9<64,64,4,128,false,2,false,true,false,false,true>;
            cudaFuncSetAttribute(bn64_nc_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_check);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, bn64_nc_k, 128, smem_check);
            printf("BM64/BN64 noncausal occupancy: %d blk/SM\n", numBlocks);
        }
        {
            constexpr size_t smem_3s = 3 * smem_check;  // 101,376 bytes triple-buffer
            auto pipeline_k = flash_attention_v9<128,64,8,128,false,1,false,true,false,false,false,true>;
            cudaFuncSetAttribute(pipeline_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_3s);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, pipeline_k, 256, smem_3s);
            printf("BM128 3-stage pipeline occupancy: %d blk/SM\n\n", numBlocks);
        }
    }

    // Correctness check
    printf("Correctness check (seq=256, 3blk, K-swizzle)...\n");
    {
        int seq_len = 256, num_heads = 32;
        size_t sz = (size_t)batch_size * num_heads * seq_len * head_dim * sizeof(half);
        half *h_Q=(half*)malloc(sz), *h_K=(half*)malloc(sz), *h_V=(half*)malloc(sz);
        half *h_O=(half*)malloc(sz), *h_ref=(half*)malloc(sz);
        srand(42);
        for (size_t i = 0; i < sz/sizeof(half); i++) {
            h_Q[i]=__float2half((float)rand()/RAND_MAX-0.5f);
            h_K[i]=__float2half((float)rand()/RAND_MAX-0.5f);
            h_V[i]=__float2half((float)rand()/RAND_MAX-0.5f);
        }
        cpu_attention(h_Q,h_K,h_V,h_ref,batch_size,num_heads,seq_len,head_dim,causal);
        half *d_Q,*d_K,*d_V,*d_O;
        CUDA_CHECK(cudaMalloc(&d_Q,sz)); CUDA_CHECK(cudaMalloc(&d_K,sz));
        CUDA_CHECK(cudaMalloc(&d_V,sz)); CUDA_CHECK(cudaMalloc(&d_O,sz));
        CUDA_CHECK(cudaMemcpy(d_Q,h_Q,sz,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K,h_K,sz,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V,h_V,sz,cudaMemcpyHostToDevice));
        flash_attention_v9_dispatch(d_Q,d_K,d_V,d_O,nullptr,nullptr,
                                    batch_size,num_heads,seq_len,head_dim,causal);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_O,d_O,sz,cudaMemcpyDeviceToHost));
        float maxdiff = 0;
        for (size_t i = 0; i < sz/sizeof(half); i++)
            maxdiff = fmaxf(maxdiff, fabsf(__half2float(h_O[i])-__half2float(h_ref[i])));
        printf("  max_diff=%.6f %s\n", maxdiff, maxdiff < 0.1f ? "PASS" : "FAIL");
        free(h_Q); free(h_K); free(h_V); free(h_O); free(h_ref);
        CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    }

    printf("Correctness check (seq=1024, causal)...\n");
    {
        int seq_len = 1024, num_heads = 32;
        size_t sz = (size_t)batch_size * num_heads * seq_len * head_dim * sizeof(half);
        half *h_Q=(half*)malloc(sz), *h_K=(half*)malloc(sz), *h_V=(half*)malloc(sz);
        half *h_O=(half*)malloc(sz), *h_ref=(half*)malloc(sz);
        srand(42);
        for (size_t i = 0; i < sz/sizeof(half); i++) {
            h_Q[i]=__float2half((float)rand()/RAND_MAX-0.5f);
            h_K[i]=__float2half((float)rand()/RAND_MAX-0.5f);
            h_V[i]=__float2half((float)rand()/RAND_MAX-0.5f);
        }
        cpu_attention(h_Q,h_K,h_V,h_ref,batch_size,num_heads,seq_len,head_dim,causal);
        half *d_Q,*d_K,*d_V,*d_O;
        CUDA_CHECK(cudaMalloc(&d_Q,sz)); CUDA_CHECK(cudaMalloc(&d_K,sz));
        CUDA_CHECK(cudaMalloc(&d_V,sz)); CUDA_CHECK(cudaMalloc(&d_O,sz));
        CUDA_CHECK(cudaMemcpy(d_Q,h_Q,sz,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K,h_K,sz,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V,h_V,sz,cudaMemcpyHostToDevice));
        flash_attention_v9_dispatch(d_Q,d_K,d_V,d_O,nullptr,nullptr,
                                    batch_size,num_heads,seq_len,head_dim,causal);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_O,d_O,sz,cudaMemcpyDeviceToHost));
        float maxdiff = 0;
        for (size_t i = 0; i < sz/sizeof(half); i++)
            maxdiff = fmaxf(maxdiff, fabsf(__half2float(h_O[i])-__half2float(h_ref[i])));
        printf("  max_diff=%.6f %s\n", maxdiff, maxdiff < 0.1f ? "PASS" : "FAIL");
        free(h_Q); free(h_K); free(h_V); free(h_O); free(h_ref);
        CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    }

    printf("Correctness check (seq=1024, non-causal)...\n");
    {
        int seq_len = 1024, num_heads = 32;
        size_t sz = (size_t)batch_size * num_heads * seq_len * head_dim * sizeof(half);
        half *h_Q=(half*)malloc(sz), *h_K=(half*)malloc(sz), *h_V=(half*)malloc(sz);
        half *h_O=(half*)malloc(sz), *h_ref=(half*)malloc(sz);
        srand(42);
        for (size_t i = 0; i < sz/sizeof(half); i++) {
            h_Q[i]=__float2half((float)rand()/RAND_MAX-0.5f);
            h_K[i]=__float2half((float)rand()/RAND_MAX-0.5f);
            h_V[i]=__float2half((float)rand()/RAND_MAX-0.5f);
        }
        cpu_attention(h_Q,h_K,h_V,h_ref,batch_size,num_heads,seq_len,head_dim,false);
        half *d_Q,*d_K,*d_V,*d_O;
        CUDA_CHECK(cudaMalloc(&d_Q,sz)); CUDA_CHECK(cudaMalloc(&d_K,sz));
        CUDA_CHECK(cudaMalloc(&d_V,sz)); CUDA_CHECK(cudaMalloc(&d_O,sz));
        CUDA_CHECK(cudaMemcpy(d_Q,h_Q,sz,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K,h_K,sz,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V,h_V,sz,cudaMemcpyHostToDevice));
        flash_attention_v9_dispatch(d_Q,d_K,d_V,d_O,nullptr,nullptr,
                                    batch_size,num_heads,seq_len,head_dim,false);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_O,d_O,sz,cudaMemcpyDeviceToHost));
        float maxdiff = 0;
        for (size_t i = 0; i < sz/sizeof(half); i++)
            maxdiff = fmaxf(maxdiff, fabsf(__half2float(h_O[i])-__half2float(h_ref[i])));
        printf("  max_diff=%.6f %s\n", maxdiff, maxdiff < 0.1f ? "PASS" : "FAIL");
        free(h_Q); free(h_K); free(h_V); free(h_O); free(h_ref);
        CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    }

    printf("Correctness check (seq=2048, non-causal, BN128)...\n");
    {
        int seq_len = 2048, num_heads = 2;
        size_t sz = (size_t)batch_size * num_heads * seq_len * head_dim * sizeof(half);
        half *h_Q=(half*)malloc(sz), *h_K=(half*)malloc(sz), *h_V=(half*)malloc(sz);
        half *h_O=(half*)malloc(sz), *h_ref=(half*)malloc(sz);
        srand(42);
        for (size_t i = 0; i < sz/sizeof(half); i++) {
            h_Q[i]=__float2half((float)rand()/RAND_MAX-0.5f);
            h_K[i]=__float2half((float)rand()/RAND_MAX-0.5f);
            h_V[i]=__float2half((float)rand()/RAND_MAX-0.5f);
        }
        cpu_attention(h_Q,h_K,h_V,h_ref,batch_size,num_heads,seq_len,head_dim,false);
        half *d_Q,*d_K,*d_V,*d_O;
        CUDA_CHECK(cudaMalloc(&d_Q,sz)); CUDA_CHECK(cudaMalloc(&d_K,sz));
        CUDA_CHECK(cudaMalloc(&d_V,sz)); CUDA_CHECK(cudaMalloc(&d_O,sz));
        CUDA_CHECK(cudaMemcpy(d_Q,h_Q,sz,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K,h_K,sz,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V,h_V,sz,cudaMemcpyHostToDevice));
        flash_attention_v9_dispatch(d_Q,d_K,d_V,d_O,nullptr,nullptr,
                                    batch_size,num_heads,seq_len,head_dim,false);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_O,d_O,sz,cudaMemcpyDeviceToHost));
        float maxdiff = 0;
        for (size_t i = 0; i < sz/sizeof(half); i++)
            maxdiff = fmaxf(maxdiff, fabsf(__half2float(h_O[i])-__half2float(h_ref[i])));
        printf("  max_diff=%.6f %s\n\n", maxdiff, maxdiff < 0.1f ? "PASS" : "FAIL");
        free(h_Q); free(h_K); free(h_V); free(h_O); free(h_ref);
        CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    }

    // Benchmark both causal and non-causal
    struct BenchConfig { int seq_len; int num_heads; };
    BenchConfig configs[] = {
        {512,   32},
        {768,   32},
        {1024,  32},
        {2048,  32},
        {4096,  32},
        {8192,  32},
        {16384, 32},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);
    const int NUM_RUNS = 3;

    for (int pass = 0; pass < 2; pass++) {
        bool bench_causal = (pass == 1);  // non-causal first (cool GPU), causal second
        if (pass > 0) { printf("\nCooldown 5s...\n"); CUDA_CHECK(cudaDeviceSynchronize()); usleep(5000000); }
        printf("\n=== %s ===\n", bench_causal ? "CAUSAL" : "NON-CAUSAL");
        printf("%-6s  %-5s  %-6s  Run1    Run2    Run3    Avg\n", "seq", "heads", "Vmode");
        printf("------------------------------------------------------\n");

        for (int t = 0; t < num_configs; t++) {
            int seq_len = configs[t].seq_len;
            int num_heads = configs[t].num_heads;
            const char* vmode;
            if (bench_causal) {
                vmode = (seq_len >= 2048) ? "128/1b" : "64/2b";
            } else {
                vmode = (seq_len >= 2048) ? "128bn" : "64/2b";
            }

            size_t sz = (size_t)batch_size * num_heads * seq_len * head_dim * sizeof(half);
            if (sz * 4 > 15ULL*1024*1024*1024) {
                printf("%-6d  %-5d  %-6s  SKIP\n", seq_len, num_heads, vmode);
                continue;
            }

            half *h_Q=(half*)malloc(sz), *h_K=(half*)malloc(sz), *h_V=(half*)malloc(sz);
            srand(42);
            for (size_t i = 0; i < sz/sizeof(half); i++) {
                h_Q[i]=__float2half((float)rand()/RAND_MAX-0.5f);
                h_K[i]=__float2half((float)rand()/RAND_MAX-0.5f);
                h_V[i]=__float2half((float)rand()/RAND_MAX-0.5f);
            }

            half *d_Q,*d_K,*d_V,*d_O;
            CUDA_CHECK(cudaMalloc(&d_Q,sz)); CUDA_CHECK(cudaMalloc(&d_K,sz));
            CUDA_CHECK(cudaMalloc(&d_V,sz)); CUDA_CHECK(cudaMalloc(&d_O,sz));
            CUDA_CHECK(cudaMemcpy(d_Q,h_Q,sz,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_K,h_K,sz,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_V,h_V,sz,cudaMemcpyHostToDevice));

            long long flops = 4LL * batch_size * num_heads * seq_len * seq_len * head_dim;
            if (bench_causal) flops /= 2;

            float runs[NUM_RUNS], sum = 0;
            for (int r = 0; r < NUM_RUNS; r++) {
                if (r > 0) { CUDA_CHECK(cudaDeviceSynchronize()); usleep(1000000); }
                for (int i = 0; i < 20; i++)
                    flash_attention_v9_dispatch(d_Q,d_K,d_V,d_O,nullptr,nullptr,
                                                batch_size,num_heads,seq_len,head_dim,bench_causal);
                CUDA_CHECK(cudaDeviceSynchronize());

                cudaEvent_t start, stop;
                CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
                CUDA_CHECK(cudaEventRecord(start));
                for (int i = 0; i < 100; i++)
                    flash_attention_v9_dispatch(d_Q,d_K,d_V,d_O,nullptr,nullptr,
                                                batch_size,num_heads,seq_len,head_dim,bench_causal);
                CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
                float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); ms /= 100;
                runs[r] = (flops / (ms / 1000.0f)) / 1e12f;
                sum += runs[r];
                CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
            }

            float avg = sum / NUM_RUNS;
            printf("%-6d  %-5d  %-6s  %5.2f   %5.2f   %5.2f   %5.2f\n",
                   seq_len, num_heads, vmode,
                   runs[0], runs[1], runs[2], avg);

            free(h_Q); free(h_K); free(h_V);
            CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
            CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
        }
    }

    return 0;
}
