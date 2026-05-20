import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter

# =========================
# Data
# =========================
labels = ["Budget=1%", "Budget=5%", "Budget=10%", "Budget=20%"]

memory_overhead = [1.3, 2.5, 4.9, 9.9]
latency_overhead = [1.0, 2.2, 4.3, 8.7]

colors = [
    "#cf475d",  # red
    "#f08a57",  # orange
    "#f2c54f",  # yellow
    "#d9b091",  # light brown
]

# =========================
# Style
# =========================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 16,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.linewidth": 1.0,
})

def percent_formatter(x, pos):
    return f"{int(x)}%"

# =========================
# Figure and Axes
# =========================
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), dpi=300)

x = np.arange(len(labels))
bar_width = 0.72

plot_data = [
    (memory_overhead, "Memory Overhead"),
    (latency_overhead, "Latency Overhead")
]

for ax, (values, ylabel) in zip(axes, plot_data):
    bars = ax.bar(
        x,
        values,
        width=bar_width,
        color=colors,
        edgecolor="#555555",
        linewidth=1.0,
        zorder=3
    )

    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_ylim(0, 11)
    ax.set_yticks([0, 2.5, 5, 7.5, 10])
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

    ax.grid(axis="y", linestyle="-", linewidth=0.8, alpha=0.2, zorder=0)

    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(1.0)

    ax.tick_params(axis="y", length=3, width=1.0, colors="#333333")

    # Value labels inside bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.35,
            f"{value:.1f}%",
            ha="center",
            va="top",
            fontsize=14,
            color="#7a7a7a",
            path_effects=[
                pe.withStroke(linewidth=1.2, foreground="white", alpha=0.9)
            ]
        )

# =========================
# Shared Legend
# =========================
handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor="#555555", linewidth=1.0)
    for i in range(len(labels))
]

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=True,
    fancybox=True,
    framealpha=1.0,
    edgecolor="#999999",
    bbox_to_anchor=(0.5, 1.03),
    columnspacing=1.1,
    handlelength=1.1,
)

plt.subplots_adjust(
    top=0.78,
    bottom=0.12,
    left=0.12,
    right=0.98,
    wspace=0.34
)

# Save if needed
plt.savefig("budget_overhead.png", dpi=300, bbox_inches="tight")
plt.savefig("budget_overhead.pdf", bbox_inches="tight")

plt.show()