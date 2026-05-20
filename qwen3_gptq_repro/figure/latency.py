import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -----------------------------
# Data approximated from the figure
# -----------------------------
batch_labels = ["bz=1", "bz=2", "bz=4", "bz=8"]

methods = ["TensorRT-LLM", "AWQ", "QuaRot", "FlatQuant", "TwinQuant"]
colors = {
    "TensorRT-LLM": "#b73a49",
    "AWQ": "#dd7b5a",
    "QuaRot": "#efc47e",
    "FlatQuant": "#86b49c",
    "TwinQuant": "#30354f",
}

env1 = {
    "TensorRT-LLM": [62, 120, 230, 420],
    "AWQ":         [55, 118, 190, 405],
    "QuaRot":      [0,  32,  72,  210],
    "FlatQuant":   [72, 125, 238, 375],
    "TwinQuant":   [126, 253, 412, 823],
}

env2 = {
    "TensorRT-LLM": [48,  98, 190, 375],
    "AWQ":         [44,  88, 172, 340],
    "QuaRot":      [0,   0,  45,  195],
    "FlatQuant":   [58, 105, 205, 285],
    "TwinQuant":   [82, 146, 238, 622],
}

speedup_env1 = ["2.04×", "2.11×", "1.79×", "1.96×"]
speedup_env2 = ["1.71×", "1.49×", "1.25×", "1.66×"]


# -----------------------------
# Helper function
# -----------------------------
def plot_broken_bar(ax_top, ax_bottom, data, title, speedups,
                    lower_ylim, upper_ylim,
                    lower_yticks, upper_yticks):
    x = np.arange(len(batch_labels))
    width = 0.16
    offsets = np.linspace(-2, 2, len(methods)) * width

    for i, method in enumerate(methods):
        y = np.array(data[method])

        ax_bottom.bar(
            x + offsets[i], y, width,
            label=method,
            color=colors[method],
            edgecolor="#4d4d4d",
            linewidth=1.0,
            zorder=3,
        )
        ax_top.bar(
            x + offsets[i], y, width,
            color=colors[method],
            edgecolor="#4d4d4d",
            linewidth=1.0,
            zorder=3,
        )

    # Axis limits
    ax_bottom.set_ylim(*lower_ylim)
    ax_top.set_ylim(*upper_ylim)

    # Ticks
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(batch_labels, fontsize=13)
    ax_bottom.set_yticks(lower_yticks)
    ax_top.set_yticks(upper_yticks)

    # Title
    ax_top.set_title(title, fontsize=14, pad=4)

    # Grid
    for ax in [ax_top, ax_bottom]:
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
        ax.tick_params(axis="both", labelsize=12, width=1.0, length=3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#777777")

    # Hide the spines between upper and lower axes
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)

    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bottom.tick_params(axis="x", top=False)

    # Broken-axis marks
    d = 0.012
    kwargs_top = dict(transform=ax_top.transAxes, color="#777777", clip_on=False, linewidth=1.5)
    kwargs_bottom = dict(transform=ax_bottom.transAxes, color="#777777", clip_on=False, linewidth=1.5)

    # left break mark
    ax_top.plot((-d, +d), (-d, +d), **kwargs_top)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs_bottom)

    # Speedup annotations above TwinQuant bars
    twin_vals = np.array(data["TwinQuant"])
    twin_x = x + offsets[methods.index("TwinQuant")]

    for xi, yi, txt in zip(twin_x, twin_vals, speedups):
        target_ax = ax_top if yi >= upper_ylim[0] else ax_bottom
        target_ax.text(
            xi,
            yi + (upper_ylim[1] - upper_ylim[0]) * 0.03 if target_ax is ax_top else yi + 10,
            txt,
            ha="center",
            va="bottom",
            fontsize=12,
            color="#ff4a4a",
            fontweight="bold",
        )


# -----------------------------
# Figure layout
# -----------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.linewidth"] = 1.0

fig = plt.figure(figsize=(12, 4.2))
gs = GridSpec(
    2, 2,
    height_ratios=[1.05, 2.15],
    width_ratios=[1, 1],
    hspace=0.04,
    wspace=0.10,
)

ax1_top = fig.add_subplot(gs[0, 0])
ax1_bottom = fig.add_subplot(gs[1, 0], sharex=ax1_top)

ax2_top = fig.add_subplot(gs[0, 1])
ax2_bottom = fig.add_subplot(gs[1, 1], sharex=ax2_top)

plot_broken_bar(
    ax1_top,
    ax1_bottom,
    env1,
    title="Environment 1",
    speedups=speedup_env1,
    lower_ylim=(0, 430),
    upper_ylim=(780, 835),
    lower_yticks=[0, 100, 200, 300, 400],
    upper_yticks=[795, 810, 825],
)

plot_broken_bar(
    ax2_top,
    ax2_bottom,
    env2,
    title="Environment 2",
    speedups=speedup_env2,
    lower_ylim=(0, 370),
    upper_ylim=(570, 635),
    lower_yticks=[0, 80, 160, 240, 320],
    upper_yticks=[580, 600, 620],
)

# Y label only on the left side
ax1_bottom.set_ylabel("Throughput (tokens/s)", fontsize=14, labelpad=2)
ax1_top.set_ylabel("")

# Remove repeated y labels on the right subplot
ax2_top.set_ylabel("")
ax2_bottom.set_ylabel("")

# Legend
handles, labels = ax1_bottom.get_legend_handles_labels()
legend = fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(0.5, 1.04),
    frameon=True,
    fontsize=13,
)
legend.get_frame().set_edgecolor("#dddddd")
legend.get_frame().set_linewidth(1.0)

# Layout adjustment
fig.subplots_adjust(top=0.84, left=0.07, right=0.995, bottom=0.12)

# Save and show
out_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, "throughput_broken_axis.pdf")
png_path = os.path.join(out_dir, "throughput_broken_axis.png")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.show()
