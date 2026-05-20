import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter

# =========================
# Data
# =========================
models = [
    "LLaMA3-3B",
    "LLaMA3-8B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
]

memory_overhead = [8.8, 8.7, 9.0, 8.5, 8.9, 9.5]
latency_overhead = [5.9, 5.3, 5.8, 5.6, 5.4, 6.2]

colors = [
    "#cf475d",  # LLaMA3-3B
    "#f1855a",  # LLaMA3-8B
    "#ffcc55",  # Qwen3-4B
    "#e6b797",  # Qwen3-8B
    "#cc7652",  # Qwen3-14B
    "#96604d",  # Qwen3-32B
]

# =========================
# Visual knobs
# =========================
TEXT_COLOR = "#000000"
FONT_SIZE_SCALE = 1.0
FONT_SIZES = {
    "base": 16 * FONT_SIZE_SCALE,
    "axis_label": 18 * FONT_SIZE_SCALE,
    "xtick": 16 * FONT_SIZE_SCALE,
    "ytick": 16 * FONT_SIZE_SCALE,
    "legend": 18 * FONT_SIZE_SCALE,
    "value_label": 14 * FONT_SIZE_SCALE,
}
Y_LIM = (0, 10.5)
Y_TICKS = [0, 2, 4, 6, 8, 10]

# =========================
# Style
# =========================
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": FONT_SIZES["base"],
    "axes.labelsize": FONT_SIZES["axis_label"],
    "xtick.labelsize": FONT_SIZES["xtick"],
    "ytick.labelsize": FONT_SIZES["ytick"],
    "legend.fontsize": FONT_SIZES["legend"],
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "axes.linewidth": 1.0,
})

def percent_formatter(x, pos):
    return f"{int(x)}%"

# =========================
# Plot
# =========================
fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

x = np.arange(len(models))
bar_width = 0.78

for ax, values, ylabel in zip(
    axes,
    [memory_overhead, latency_overhead],
    ["Memory Overhead", "Latency Overhead"]
):
    bars = ax.bar(
        x,
        values,
        width=bar_width,
        color=colors,
        edgecolor="#666666",
        linewidth=0.8,
        zorder=3,
    )

    ax.set_ylabel(ylabel, fontsize=FONT_SIZES["axis_label"], color=TEXT_COLOR)
    ax.set_ylim(*Y_LIM)
    ax.set_yticks(Y_TICKS)
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

    ax.set_xticks([])
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.25, zorder=0)

    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(1.0)

    ax.tick_params(axis="y", length=3, width=1.0,
                   labelsize=FONT_SIZES["ytick"], colors=TEXT_COLOR)

    # Value labels inside bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.25,
            f"{value:.1f}%",
            ha="center",
            va="top",
            fontsize=FONT_SIZES["value_label"],
            color=TEXT_COLOR,
            path_effects=[
                pe.withStroke(linewidth=1.2, foreground="white", alpha=0.8)
            ],
        )

# =========================
# Shared legend
# =========================
handles = [
    plt.Rectangle(
        (0, 0), 1, 1,
        facecolor=colors[i],
        edgecolor="#666666",
        linewidth=0.8
    )
    for i in range(len(models))
]

legend = fig.legend(
    handles,
    models,
    loc="upper center",
    ncol=3,
    frameon=True,
    fancybox=True,
    framealpha=1.0,
    edgecolor="#888888",
    bbox_to_anchor=(0.5, 0.93),
    columnspacing=1.2,
    handlelength=1.1,
    fontsize=FONT_SIZES["legend"],
)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)

plt.subplots_adjust(
    top=0.72,
    bottom=0.14,
    left=0.10,
    right=0.98,
    wspace=0.28,
)

# =========================
# Save
# =========================
out_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, "low_rank_overhead.pdf")
png_path = os.path.join(out_dir, "low_rank_overhead.png")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, bbox_inches="tight", dpi=300)

plt.show()
