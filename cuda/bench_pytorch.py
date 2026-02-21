#!/usr/bin/env python3
"""
Benchmark PyTorch Flash Attention (sdpa_flash backend) across sequence lengths.

Measures torch.nn.functional.scaled_dot_product_attention with the flash backend
forced via torch.backends.cuda.sdp_kernel context manager.

Reports mean/min/max TFLOPS for each sequence length over 3 independent runs,
each consisting of 20 warmup + 100 timed iterations.
"""

import torch
import torch.nn.functional as F


def compute_causal_flops(batch: int, heads: int, seq: int, head_dim: int) -> float:
    """
    TFLOPS-relevant FLOPs for causal flash attention.
    flops = 4 * batch * heads * seq * seq * head_dim / 2   (the /2 accounts for causal masking)
    """
    return 4 * batch * heads * seq * seq * head_dim / 2


def bench_one(
    batch: int,
    heads: int,
    seq: int,
    head_dim: int,
    n_warmup: int = 20,
    n_iters: int = 100,
) -> float:
    """
    Run one measurement: n_warmup warmup iterations followed by n_iters timed
    iterations.  Returns elapsed time in seconds for the timed portion.
    """
    device = torch.device("cuda")
    q = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16, device=device)

    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        # --- warmup ---
        for _ in range(n_warmup):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()

        # --- timed iterations ---
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(n_iters):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        end_event.record()

        torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)  # milliseconds
    return elapsed_ms / 1000.0  # seconds


def main() -> None:
    # ----- configuration -----
    batch = 1
    heads = 32
    head_dim = 128
    seq_lens = [256, 512, 768, 1024, 2048, 4096, 8192, 16384]
    n_warmup = 20
    n_iters = 100
    n_runs = 3  # independent measurements per seq_len

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU           : {device_name}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"batch={batch}  heads={heads}  head_dim={head_dim}  causal=True")
    print(f"warmup={n_warmup}  iters={n_iters}  runs={n_runs}")
    print("-" * 80)
    print(
        f"{'seq_len':>8}  {'mean TFLOPS':>12}  {'min TFLOPS':>11}  {'max TFLOPS':>11}  "
        f"{'mean ms/iter':>13}"
    )
    print("-" * 80)

    for seq in seq_lens:
        flops = compute_causal_flops(batch, heads, seq, head_dim)
        tflops_list = []

        for run_idx in range(n_runs):
            elapsed_s = bench_one(batch, heads, seq, head_dim, n_warmup, n_iters)
            per_iter_s = elapsed_s / n_iters
            tflops = flops / per_iter_s / 1e12
            tflops_list.append(tflops)

        mean_tflops = sum(tflops_list) / len(tflops_list)
        min_tflops = min(tflops_list)
        max_tflops = max(tflops_list)
        # mean ms per iteration (average over runs)
        mean_ms = flops / (mean_tflops * 1e12) * 1e3

        print(
            f"{seq:>8}  {mean_tflops:>12.2f}  {min_tflops:>11.2f}  {max_tflops:>11.2f}  "
            f"{mean_ms:>13.4f}"
        )

    print("-" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
