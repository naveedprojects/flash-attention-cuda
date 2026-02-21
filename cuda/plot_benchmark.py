#!/usr/bin/env python3
"""Generate Flash Attention V9 vs PyTorch FA2 benchmark comparison graphs."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ============================================================================
# Benchmark data: batch=1, heads=32, head_dim=128, RTX 3080 Ti Laptop (SM86)
# ============================================================================
seq_lens = [512, 768, 1024, 2048, 4096, 8192, 16384]

# V9 Causal - best numbers (15s cooldown run, S/P merge)
v9_causal = [19.72, 22.91, 23.56, 24.33, 26.91, 27.84, 26.62]

# V9 Non-Causal - best numbers (15s cooldown run, S/P merge, BN=128 for >=2048)
v9_noncausal = [21.72, 24.86, 25.78, 29.62, 30.09, 30.37, 30.20]

# PyTorch FA2 Causal (sdpa_flash, 3-run avg)
pt_causal = [17.25, 19.56, 20.26, 22.19, 21.87, 20.66, 18.11]

# PyTorch FA2 Non-Causal (sdpa_flash, 3-run avg)
pt_noncausal = [20.35, 20.43, 30.20, 29.98, 29.86, 23.52, 18.49]

# ============================================================================
# Plot
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

# Colors
v9_color = '#2563eb'    # blue
pt_color = '#dc2626'    # red
marker_v9 = 'o'
marker_pt = 's'
lw = 2.2
ms = 7

def format_ax(ax, title, data_pairs):
    """Configure axis with consistent styling."""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('TFLOPS (FP16)', fontsize=11)
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lens)
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{int(x)}' if x < 1024 else f'{int(x/1024)}K'))
    ax.set_xlim(400, 20000)

    # Y range from data
    all_vals = [v for pair in data_pairs for v in pair]
    ymin = max(0, min(all_vals) - 3)
    ymax = max(all_vals) + 3
    ax.set_ylim(ymin, ymax)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)

# --- Causal subplot ---
ax1.plot(seq_lens, v9_causal, color=v9_color, marker=marker_v9,
         linewidth=lw, markersize=ms, label='Flash Attention V9 (Ours)', zorder=5)
ax1.plot(seq_lens, pt_causal, color=pt_color, marker=marker_pt,
         linewidth=lw, markersize=ms, label='PyTorch FA2 (Tri Dao)', zorder=5)

# Shade the winning region
for i in range(len(seq_lens) - 1):
    if v9_causal[i] >= pt_causal[i] and v9_causal[i+1] >= pt_causal[i+1]:
        ax1.fill_between(seq_lens[i:i+2], pt_causal[i:i+2], v9_causal[i:i+2],
                         alpha=0.08, color=v9_color)

format_ax(ax1, 'Causal Attention', [v9_causal, pt_causal])

# Add ratio annotations for causal
for i, seq in enumerate(seq_lens):
    ratio = v9_causal[i] / pt_causal[i] * 100
    if ratio >= 100:
        ax1.annotate(f'+{ratio-100:.0f}%', (seq, v9_causal[i]),
                     textcoords='offset points', xytext=(0, 10),
                     fontsize=7.5, ha='center', color=v9_color, fontweight='bold')

# --- Non-Causal subplot ---
ax2.plot(seq_lens, v9_noncausal, color=v9_color, marker=marker_v9,
         linewidth=lw, markersize=ms, label='Flash Attention V9 (Ours)', zorder=5)
ax2.plot(seq_lens, pt_noncausal, color=pt_color, marker=marker_pt,
         linewidth=lw, markersize=ms, label='PyTorch FA2 (Tri Dao)', zorder=5)

# Shade winning regions
for i in range(len(seq_lens) - 1):
    if v9_noncausal[i] >= pt_noncausal[i] and v9_noncausal[i+1] >= pt_noncausal[i+1]:
        ax2.fill_between(seq_lens[i:i+2], pt_noncausal[i:i+2], v9_noncausal[i:i+2],
                         alpha=0.08, color=v9_color)

format_ax(ax2, 'Non-Causal Attention', [v9_noncausal, pt_noncausal])

# Add ratio annotations for non-causal
for i, seq in enumerate(seq_lens):
    ratio = v9_noncausal[i] / pt_noncausal[i] * 100
    if ratio >= 100:
        ax2.annotate(f'+{ratio-100:.0f}%', (seq, v9_noncausal[i]),
                     textcoords='offset points', xytext=(0, 10),
                     fontsize=7.5, ha='center', color=v9_color, fontweight='bold')

# Main title
fig.suptitle('Flash Attention V9 vs PyTorch FlashAttention-2  —  FP16 Forward Pass\n'
             'RTX 3080 Ti Laptop  |  batch=1, heads=32, head_dim=128',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/home/naveed/Documents/self/relatch/flash-attention-cuda/benchmark_results.png',
            bbox_inches='tight', pad_inches=0.2)
print('Saved: benchmark_results.png')
plt.close()
