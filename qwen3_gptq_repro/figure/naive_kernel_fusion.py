#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latency breakdown figure: Naive vs Opt (W4A4 sparse-augmented GEMM).

Vertical stacked bars, Nunchaku-style reference layout:
  - full box frame, horizontal gridlines only
  - white bold in-bar numbers (moved outside when segment is too small)
  - dashed red drop-line from naive top to opt top, centred arrow + NNNx label
  - serif legend on the right, italic (a)-prefixed caption below

Palette matches docs/figures/pipeline_spacetime/naive_pipeline.tex.

Usage:
    python3 plot_latency_breakdown.py
Outputs latency_breakdown.pdf (vector, for LaTeX) and latency_breakdown.png.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ----------------------------------------------------------------------------
# Nature-compatible high-contrast palette (Okabe-Ito inspired).
# ----------------------------------------------------------------------------
qC   = "#6E6E6E"      # quant
tcC  = "#0072B2"      # dense GEMM  (INT4 Tensor Core)
spC  = "#009E73"      # sparse branch
redC = "#D55E00"      # reduce-add
fpC  = "#CC79A7"      # fused (dense+sparse)
rtC  = "#C00000"      # speed-up annotation
axisC = "#1A1A1A"

# ----------------------------------------------------------------------------
# Raw data  (microseconds)  -- from fig/data.md
# ----------------------------------------------------------------------------
NAIVE = [
    ("Quant",         14.24, qC),
    ("Dense GEMM",    29.78, tcC),
    ("Sparse GEMM",   16.49, spC),
    ("Reduce Add",     5.83, redC),
]
OPT = [
    ("Quant",                17.82, qC),
    ("Fused (dense+sparse)", 17.08, fpC),
]
NAIVE_TOTAL = sum(v for _, v, _ in NAIVE)   # 66.34
OPT_TOTAL   = sum(v for _, v, _ in OPT)     # 35.00
SPEEDUP     = NAIVE_TOTAL / OPT_TOTAL       # ~1.90x
# ----------------------------------------------------------------------------
# Matplotlib global style  (paper-grade serif)
# ----------------------------------------------------------------------------
rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Palatino", "Palatino Linotype", "Times New Roman",
                           "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "axes.edgecolor":     axisC,
    "axes.labelcolor":    axisC,
    "xtick.color":        axisC,
    "ytick.color":        axisC,
    "axes.linewidth":     0.9,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))

BAR_W   = 0.52                                # narrow bars, wide gap
X_NAIVE = 0.0
X_OPT   = 1.0

# Visual knobs.
BLOCK_EDGE_COLOR = "black"
BLOCK_EDGE_WIDTH = 0.9
TEXT_FONT_FAMILY = "serif"
SPEEDUP_DASH_EXTEND = 0.22
SPEEDUP_LINE_WIDTH = 2.0
FONT_SIZE_SCALE = 1.0
FONT_SIZES = {
    "segment": 12 * FONT_SIZE_SCALE,
    "speedup": 12 * FONT_SIZE_SCALE,
    "axis_label": 16 * FONT_SIZE_SCALE,
    "tick_x": 16 * FONT_SIZE_SCALE,
    "tick_y": 12 * FONT_SIZE_SCALE,
    "legend": 12 * FONT_SIZE_SCALE,
}


def _draw_stack(x: float, segments: list[tuple[str, float, str]]) -> float:
    """Stack `segments` vertically at column `x`; return running height."""
    y = 0.0
    for _, val, color in segments:
        ax.bar(x, val, bottom=y, width=BAR_W,
               color=color, edgecolor=BLOCK_EDGE_COLOR,
               linewidth=BLOCK_EDGE_WIDTH, zorder=3)
        ax.text(x, y + val / 2, f"{val:.1f}",
                ha="center", va="center",
                fontsize=FONT_SIZES["segment"],
                color="white", fontweight="bold",
                fontfamily=TEXT_FONT_FAMILY,
                zorder=5)
        y += val
    return y


naive_top = _draw_stack(X_NAIVE, NAIVE)
opt_top   = _draw_stack(X_OPT,   OPT)

# ----------------------------------------------------------------------------
# Speed-up annotation : dashed red line from naive-top right edge to opt-top,
# then a small centred arrow pointing down + "N.NNx" label.
# ----------------------------------------------------------------------------
drop_line_y = naive_top                                  # same height as naive
line_x0     = X_NAIVE + BAR_W / 2                        # right edge of naive
line_x1     = X_OPT + BAR_W / 2 + SPEEDUP_DASH_EXTEND    # extend past opt bar
ax.plot([line_x0, line_x1], [drop_line_y, drop_line_y],
        color=rtC, lw=SPEEDUP_LINE_WIDTH,
        linestyle=(0, (5, 2.5)), zorder=4)

# centred drop arrow on top of opt bar
ax.annotate(
    "", xy=(X_OPT, opt_top + 1.0),
    xytext=(X_OPT, drop_line_y - 1.0),
    arrowprops=dict(arrowstyle="-|>", color=rtC,
                    lw=SPEEDUP_LINE_WIDTH, mutation_scale=14),
    zorder=5,
)
ax.text(X_OPT + 0.02, (drop_line_y + opt_top) / 2,
        f"{SPEEDUP:.1f}x",
        ha="left", va="center", fontsize=FONT_SIZES["speedup"], color=rtC,
        fontweight="bold", fontfamily=TEXT_FONT_FAMILY, zorder=6)

# ----------------------------------------------------------------------------
# Axes cosmetics
# ----------------------------------------------------------------------------
ax.set_xticks([X_NAIVE, X_OPT])
ax.set_xticklabels(["Na\u00efve", "Ours"], fontsize=FONT_SIZES["tick_x"])
ax.set_ylabel(r"Latency ($\mu$s)", fontsize=FONT_SIZES["axis_label"])

# --- tightened Y range: just cover the top annotation ------------------------
y_top = naive_top * 1.13                                # ~13% head-room
ax.set_ylim(0, y_top)
ax.set_xlim(X_NAIVE - 0.75, X_OPT + 0.95)

# nice round y ticks
import numpy as np
step = 20
ax.set_yticks(np.arange(0, int(y_top) + 1, step))

# full-box frame (like Nunchaku)
for spine in ("top", "right", "bottom", "left"):
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(0.9)
    ax.spines[spine].set_color(axisC)

ax.tick_params(axis="y", which="major", length=3.5,
               labelsize=FONT_SIZES["tick_y"],
               direction="in", color=axisC)
ax.tick_params(axis="x", which="major", length=0,
               labelsize=FONT_SIZES["tick_x"], pad=4)
ax.grid(axis="y", which="major", color="#D9DEE3", lw=0.6, zorder=1)
ax.set_axisbelow(True)

# ----------------------------------------------------------------------------
# Legend (top)
# ----------------------------------------------------------------------------
legend_entries = [
    ("Reduce Add",            redC),
    ("Sparse Compute",           spC),
    ("Dense Compute",            tcC),
    ("Fused Compute",                 fpC),
    ("Quantize",                 qC),
]
handles = [mpatches.Patch(facecolor=c, edgecolor=BLOCK_EDGE_COLOR,
                          linewidth=BLOCK_EDGE_WIDTH, label=lbl)
           for lbl, c in legend_entries]
ax.legend(handles=handles,
          loc="lower center", bbox_to_anchor=(0.5, 1.02),
          ncol=3, frameon=False, fontsize=FONT_SIZES["legend"],
          handlelength=1.1, handleheight=1.0, borderaxespad=0.0,
          labelspacing=0.45, columnspacing=0.9, handletextpad=0.45)

plt.tight_layout()

# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, "latency_breakdown.pdf")
png_path = os.path.join(out_dir, "latency_breakdown.png")
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
fig.savefig(png_path, bbox_inches="tight", pad_inches=0.06, dpi=300)
print(f"[ok] wrote {pdf_path}")
print(f"[ok] wrote {png_path}")
