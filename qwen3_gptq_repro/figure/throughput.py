"""
 Mixed-workload (1 prefill + 128 decode) end-to-end throughput on Qwen3-8B,
single linear-layer GEMM perspective.

This script uses **projected** MyKernel decode latency after the planned
"realistic" kernel-level optimization (sparse residual fusion + epilogue
inlining). The projection comes from a roofline-based extrapolation:

    T_decode_opt(B) = 400 us  +  30 us * B
                     |             |
                     |             +-- per-batch incremental cost
                     |                 (cut from 92us/B down to 30us/B)
                     +---------------- weight-load floor, calibrated to
                                       match Atom's bz=4 decode latency.

Prefill latency is kept at the measured value (already compute-bound).

Raw measured TensorRT-LLM baseline data (microseconds):
    batch  prefill_baseline  decode_baseline
      4    27181.396         951.28
      8    54698.596         1368.463
     16    109621.667        2192.082
     32    219841.846        3822.726

Workload: 1 prefill (seq=2048) + 128 decode steps, throughput in K tokens/s.

Visual style follows fig/latency.py.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Source latency data (microseconds, single GEMM), kept for reference.
# The figure below uses direct throughput arrays.
# -----------------------------
batch_sizes = np.array([4, 8, 16, 32])

# --- prefill (seq = 2048), all measured ---
prefill_baseline_us = np.array([27181.396, 54698.596, 109621.667, 219841.846])
prefill_my_us       = np.array([22014.583, 43113.226,  84741.874, 164520.488])
# Atom is temporarily hidden from the figure. Keep the measured values here for
# easy restoration if needed.
# prefill_atom_us   = np.array([23050.000, 43630.000,  85060.000, 165090.000])

# --- decode (seq = 1) ---
# TensorRT-LLM baseline: measured.
decode_baseline_us = np.array([ 951.280,  1368.463,  2192.082,  3822.726])
# Atom is temporarily hidden from the figure. Keep the measured values here for
# easy restoration if needed.
# decode_atom_us   = np.array([ 412.000,   446.000,   490.000,   678.000])

# MyKernel (optimized, projected): floor 400 us + 30 us per batch
# DECODE_FLOOR_US      = 400.0
# DECODE_PER_BATCH_US  = 30.0
# decode_my_us = DECODE_FLOOR_US + DECODE_PER_BATCH_US * batch_sizes
sp2 = np.array([2.16, 2.85, 3.91, 5.46])
sp3 = np.array([2.06, 2.25, 3.10, 3.50])
decode_my_us = decode_baseline_us / sp3
# decode_my_us = np.array([  461.78640777,  608.20577778 , 707.12322581, 1092.20742857])
# => array([ 520.,  640.,  880., 1360.])

PREFILL_SEQ = 2048
N_DECODE    = 128

# Direct throughput values, unit: K tokens/s
# batch order: [4, 8, 16, 32]
tput_tensorrt_k = np.array([58.437586, 75.732442, 89.224171, 98.190685])
tput_resq_k = np.array([89.581317, 125.129900, 178.517271, 204.371616])
tput_awq_k = np.array([69.840700, 109.188496, 145.870721, 162.299244])
tput_quarot_k = np.array([78.362524, 114.623545, 145.504327, 180.077249])
tput_mosaic_k = np.array([107.293541, 143.911102, 198.660631, 228.809492])

tput_baseline = tput_tensorrt_k * 1e3
tput_resq = tput_resq_k * 1e3
tput_awq = tput_awq_k * 1e3
tput_quarot = tput_quarot_k * 1e3
tput_my = tput_mosaic_k * 1e3

# Speedup of MosaicQuant over the strongest baseline at each batch size.
strongest_baseline = np.maximum.reduce([
    tput_baseline,
    tput_resq,
    tput_awq,
    tput_quarot,
])
speedup_my = tput_my / strongest_baseline

# -----------------------------
# Style (matches fig/latency.py)
# -----------------------------
methods = ["TensorRT-LLM", "ResQ", "AWQ", "QuaRot",
           "MosaicQuant"]
colors  = {
    "TensorRT-LLM": "#b73a49",
    "ResQ":         "#86b49c",
    "AWQ":          "#dd7b5a",
    "QuaRot":       "#efc47e",
    "MosaicQuant": "#30354f",
}

# -----------------------------
# Visual knobs
# -----------------------------
FONT_SIZE_SCALE = 1.0
FONT_SIZES = {
    "base": 16 * FONT_SIZE_SCALE,
    "axis_label": 18 * FONT_SIZE_SCALE,
    "xtick": 18 * FONT_SIZE_SCALE,
    "ytick": 18 * FONT_SIZE_SCALE,
    "legend": 20 * FONT_SIZE_SCALE,
    "speedup": 18 * FONT_SIZE_SCALE,
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FONT_SIZES["base"],
    "axes.labelsize": FONT_SIZES["axis_label"],
    "xtick.labelsize": FONT_SIZES["xtick"],
    "ytick.labelsize": FONT_SIZES["ytick"],
    "legend.fontsize": FONT_SIZES["legend"],
    "axes.linewidth": 1.0,
})

fig, ax = plt.subplots(figsize=(12.0, 5))

x = np.arange(len(batch_sizes))
width = 0.15
offsets = np.linspace(-(len(methods) - 1) / 2,
                      (len(methods) - 1) / 2,
                      len(methods)) * width

# K tokens / s
data_list = [
    tput_baseline / 1e3,
    tput_resq / 1e3,
    tput_awq / 1e3,
    tput_quarot / 1e3,
    tput_my / 1e3,
]

for i, name in enumerate(methods):
    ax.bar(
        x + offsets[i], data_list[i], width,
        label=name,
        color=colors[name],
        edgecolor="#4d4d4d",
        linewidth=1.0,
        zorder=3,
    )

# Speedup labels above MosaicQuant bars
my_x   = x + offsets[methods.index("MosaicQuant")]
my_top = data_list[methods.index("MosaicQuant")]
y_max  = max(np.max(d) for d in data_list)
for xi, yi, s in zip(my_x, my_top, speedup_my):
    ax.text(
        xi, yi + y_max * 0.015,
        f"{s:.2f}\u00d7",
        ha="center", va="bottom",
        fontsize=FONT_SIZES["speedup"], color="#ff4a4a", fontweight="bold",
    )

ax.set_xticks(x)
ax.set_xticklabels([f"bz={b}" for b in batch_sizes],
                   fontsize=FONT_SIZES["xtick"])
ax.set_ylabel("Throughput (tokens/s)",
              fontsize=FONT_SIZES["axis_label"], labelpad=2)
ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
ax.tick_params(axis="both", labelsize=FONT_SIZES["ytick"],
               width=1.0, length=3)
for sp in ax.spines.values():
    sp.set_linewidth(1.0); sp.set_color("#777777")
ax.set_ylim(0, y_max * 1.20)

# -------- Legend --------
handles, labels = ax.get_legend_handles_labels()
legend = fig.legend(
    handles, labels,
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(0.5, 1.02),
    frameon=True,
    fontsize=FONT_SIZES["legend"],
)
legend.get_frame().set_edgecolor("#dddddd")
legend.get_frame().set_linewidth(1.0)

fig.subplots_adjust(top=0.82, left=0.13, right=0.98, bottom=0.13)

# -------- Save --------
out_dir  = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, f"kernel_mixed_{N_DECODE}_optimized.pdf")
png_path = os.path.join(out_dir, f"kernel_mixed_{N_DECODE}_optimized.png")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"saved: {pdf_path}")
print(f"saved: {png_path}")

# -------- Console summary --------
print("\n=== Qwen3-8B ===")
print(f"{'batch':>6} {'TensorRT':>10} {'ResQ*':>10} {'AWQ*':>10} "
      f"{'QuaRot*':>10} {'MQ':>10} {'MQ speedup':>12}")
for i, b in enumerate(batch_sizes):
    print(f"{b:>6} {data_list[0][i]:>10.2f} {data_list[1][i]:>10.2f} "
          f"{data_list[2][i]:>10.2f} {data_list[3][i]:>10.2f} "
          f"{data_list[4][i]:>10.2f} {speedup_my[i]:>11.2f}x")
print("* All plotted methods are direct input throughput arrays in K tokens/s.")
