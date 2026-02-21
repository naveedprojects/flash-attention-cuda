#!/usr/bin/env python3
"""
Extended PyTorch Flash Attention benchmark across sequence lengths 256-131072.

Reduces heads for long sequences to fit 12GB VRAM.
Reports mean/min/max TFLOPS for each configuration over 3 independent runs,
each consisting of 20 warmup + 100 timed iterations.
"""

import torch
import torch.nn.functional as F


def compute_causal_flops(batch: int, heads: int, seq: int, head_dim: int) -> float:
    return 4 * batch * heads * seq * seq * head_dim / 2


def bench_one(
    batch: int,
    heads: int,
    seq: int,
    head_dim: int,
    n_warmup: int = 20,
    n_iters: int = 100,
    use_flash: bool = True,
) -> float:
    device = torch.device("cuda")
    q = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(batch, heads, seq, head_dim, dtype=torch.float16, device=device)

    ctx = torch.backends.cuda.sdp_kernel(
        enable_flash=use_flash,
        enable_math=False,
        enable_mem_efficient=not use_flash,
    )

    with ctx:
        for _ in range(n_warmup):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(n_iters):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        end_event.record()

        torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    return elapsed_ms / 1000.0


def main() -> None:
    batch = 1
    head_dim = 128
    n_warmup = 20
    n_iters = 100
    n_runs = 3

    # (seq_len, heads) pairs - reduce heads for long sequences to fit 12GB
    configs = [
        (256, 32),
        (512, 32),
        (768, 32),
        (1024, 32),
        (2048, 32),
        (4096, 32),
        (8192, 32),
        (16384, 16),
        (32768, 8),
        (65536, 4),
        (131072, 2),
    ]

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU           : {device_name}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"batch={batch}  head_dim={head_dim}  causal=True")
    print(f"warmup={n_warmup}  iters={n_iters}  runs={n_runs}")
    print("-" * 90)
    print(
        f"{'seq_len':>8}  {'heads':>5}  {'mean TFLOPS':>12}  {'min TFLOPS':>11}  "
        f"{'max TFLOPS':>11}  {'mean ms/iter':>13}  {'backend':>10}"
    )
    print("-" * 90)

    for seq, heads in configs:
        flops = compute_causal_flops(batch, heads, seq, head_dim)

        # Estimate VRAM: 3 tensors * batch * heads * seq * head_dim * 2 bytes
        vram_est = 3 * batch * heads * seq * head_dim * 2 / 1e9
        if vram_est > 11.0:
            print(f"{seq:>8}  {heads:>5}  {'SKIP (VRAM)':>12}")
            continue

        # Try flash first, fall back to mem_efficient
        backend = "flash"
        try:
            tflops_list = []
            for _ in range(n_runs):
                elapsed_s = bench_one(batch, heads, seq, head_dim, n_warmup, n_iters, use_flash=True)
                per_iter_s = elapsed_s / n_iters
                tflops = flops / per_iter_s / 1e12
                tflops_list.append(tflops)
        except RuntimeError:
            backend = "mem_eff"
            tflops_list = []
            for _ in range(n_runs):
                elapsed_s = bench_one(batch, heads, seq, head_dim, n_warmup, n_iters, use_flash=False)
                per_iter_s = elapsed_s / n_iters
                tflops = flops / per_iter_s / 1e12
                tflops_list.append(tflops)

        mean_tflops = sum(tflops_list) / len(tflops_list)
        min_tflops = min(tflops_list)
        max_tflops = max(tflops_list)
        mean_ms = flops / (mean_tflops * 1e12) * 1e3

        print(
            f"{seq:>8}  {heads:>5}  {mean_tflops:>12.2f}  {min_tflops:>11.2f}  "
            f"{max_tflops:>11.2f}  {mean_ms:>13.4f}  {backend:>10}"
        )

    print("-" * 90)
    print("Done.")


if __name__ == "__main__":
    main()
