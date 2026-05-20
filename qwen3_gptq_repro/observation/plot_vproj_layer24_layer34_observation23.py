#!/usr/bin/env python3
"""原论文中的图2（b）

输入:
  只读取 diagnose_hessian_sensitive_blocks_per_layer.py 生成的
  observation23_summary.json, 不重新加载模型、不重新计算 Hessian。
  默认输入是:
    output/hessian_sensitive_blocks_vproj_l0_35_smooth_v16_b32x128_g128_per_layer/
    observation23_summary.json

主要参数:
  --summary-json 指定 observation23_summary.json。
  --layers 指定两个要并排比较的层, 默认 24,34。
  --module-name 只用于输出文件名, 默认 v_proj。
  --max-fraction 控制 x 轴最大 selected block fraction, 默认 0.20。
  --format / --dpi 控制保存格式和分辨率。
  --fig-width / --fig-height 以及各 font-size 参数控制版式。

运行示例:
  cd /root/autodl-tmp/Zip/qwen3_gptq_repro
  python3 observation/plot_vproj_layer24_layer34_observation23.py \
    --summary-json observation/output/hessian_sensitive_blocks_vproj_l0_35_smooth_v16_b32x128_g128_per_layer/observation23_summary.json \
    --layers 24,34 \
    --module-name v_proj \
    --max-fraction 0.20 \
    --format both

画出的图像:
  <output-dir>/v_proj_layer24_layer34_observation23_selection_curve.png|pdf
    双面板曲线图。左/右面板分别对应 --layers 给出的两层; x 轴是选中
    block 比例, y 轴是捕获的 layer output error reduction。曲线比较
    output distortion oracle、weight magnitude、residual norm 和 random。
"""


# python3 observation/plot_vproj_layer24_layer34_observation23.py \
#   --summary-json output/hessian_sensitive_blocks_vproj_l0_35_smooth_v16_b32x128_g128_per_layer/observation23_summary.json \


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_JSON = (
    SCRIPT_DIR
    / "output"
    / "hessian_sensitive_blocks_vproj_l0_35_smooth_v16_b32x128_g128_per_layer"
    / "observation23_summary.json"
)

# Each tuple defines one curve in the final two-panel figure:
# (method key in observation23_summary.json, legend label, color, line style, width).
METHOD_STYLES = (
    ("exact_oracle", "output distortion", "black", "-", 2.2),
    ("weight_outlier", "weight magnitude", "#E69F00", "--", 2.0),
    ("residual_norm", "residual norm", "#0072B2", "-.", 2.0),
    ("random", "random", "#7A7A7A", ":", 2.2),
)


def parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            layers.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid layer id: {part!r}") from exc
    if len(layers) != 2:
        raise argparse.ArgumentTypeError("--layers must contain exactly two comma-separated ids")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot v_proj Layer 24/34 Observation 2/3 curves from observation23_summary.json."
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="Path to observation23_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated figure. Defaults to <summary-json-dir>/plots/observation23.",
    )
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default=parse_layers("24,34"),
        help="Exactly two comma-separated layer ids.",
    )
    parser.add_argument("--module-name", type=str, default="v_proj")
    parser.add_argument(
        "--max-fraction",
        type=float,
        default=0.20,
        help="Maximum selected block fraction shown on the x-axis.",
    )
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig-width", type=float, default=10)
    parser.add_argument("--fig-height", type=float, default=5)
    parser.add_argument("--title-font-size", type=float, default=22.0)
    parser.add_argument("--label-font-size", type=float, default=22.0)
    parser.add_argument("--tick-font-size", type=float, default=18.0)
    parser.add_argument("--legend-font-size", type=float, default=18)
    args = parser.parse_args()

    if not (0.0 < args.max_fraction <= 1.0):
        parser.error("--max-fraction must be in (0, 1].")
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    if args.fig_width <= 0 or args.fig_height <= 0:
        parser.error("--fig-width and --fig-height must be positive.")
    for attr in ("title_font_size", "label_font_size", "tick_font_size", "legend_font_size"):
        if getattr(args, attr) <= 0:
            parser.error(f"--{attr.replace('_', '-')} must be positive.")
    return args


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def make_output_filename_stem(module_name: str, layers: list[int]) -> str:
    clean_module_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in module_name).strip("_")
    if not clean_module_name:
        clean_module_name = "module"
    return f"{clean_module_name}_layer{layers[0]}_layer{layers[1]}_observation23_selection_curve"


def layer_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def load_layer_repair_rows(summary_json: Path, layers: list[int]) -> dict[int, list[dict]]:
    if not summary_json.exists():
        raise FileNotFoundError(f"summary JSON not found: {summary_json}")
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    per_layer = data.get("per_layer")
    if not isinstance(per_layer, dict):
        raise ValueError(f"summary JSON does not contain a per_layer object: {summary_json}")

    rows_by_layer: dict[int, list[dict]] = {}
    missing_layers: list[int] = []
    for layer_id in layers:
        layer_summary = per_layer.get(str(layer_id))
        if not isinstance(layer_summary, dict):
            missing_layers.append(layer_id)
            continue
        repair_rows = layer_summary.get("repair_rows")
        if not isinstance(repair_rows, list) or not repair_rows:
            raise ValueError(f"Layer {layer_id} has no repair_rows in {summary_json}")
        rows_by_layer[layer_id] = repair_rows

    if missing_layers:
        available = ", ".join(sorted(per_layer.keys(), key=layer_sort_key))
        raise KeyError(f"missing layers {missing_layers}; available layers: {available}")
    return rows_by_layer


def plot_method_curve(ax, repair_rows: list[dict], method: str, label: str, color: str, linestyle: str, linewidth: float):
    rows = sorted(
        (row for row in repair_rows if row.get("method") == method),
        key=lambda row: float(row.get("fraction", 0.0)),
    )
    if not rows:
        return
    x = np.array([100.0 * float(row["fraction"]) for row in rows], dtype=np.float64)
    y = np.array(
        [np.nan if row.get("captured_mass") is None else 100.0 * float(row["captured_mass"]) for row in rows],
        dtype=np.float64,
    )
    ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth)


def captured_mass_percent_at(repair_rows: list[dict], method: str, fraction: float) -> float | None:
    rows = sorted(
        (
            row
            for row in repair_rows
            if row.get("method") == method and row.get("captured_mass") is not None
        ),
        key=lambda row: float(row.get("fraction", 0.0)),
    )
    if not rows:
        return None

    x = np.array([float(row["fraction"]) for row in rows], dtype=np.float64)
    y = np.array([100.0 * float(row["captured_mass"]) for row in rows], dtype=np.float64)
    exact_matches = np.isclose(x, fraction, rtol=0.0, atol=1e-10)
    if exact_matches.any():
        return float(y[np.where(exact_matches)[0][0]])
    if fraction < float(x.min()) or fraction > float(x.max()):
        return None
    return float(np.interp(fraction, x, y))


def add_output_distortion_marker(ax, repair_rows: list[dict], fraction: float) -> None:
    y = captured_mass_percent_at(repair_rows, "exact_oracle", fraction)
    if y is None:
        return

    x = 100.0 * fraction
    ax.scatter(
        [x],
        [y],
        marker="*",
        s=170,
        color="red",
        edgecolors="white",
        linewidths=0.7,
        zorder=6,
    )


def save_two_layer_plot(args: argparse.Namespace, rows_by_layer: dict[int, list[dict]]) -> list[Path]:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(args.fig_width, args.fig_height),
        sharey=True,
        constrained_layout=True,
    )

    max_x = 100.0 * float(args.max_fraction)
    for ax, layer_id in zip(axes, args.layers):
        repair_rows = rows_by_layer[layer_id]
        for method, label, color, linestyle, linewidth in METHOD_STYLES:
            plot_method_curve(ax, repair_rows, method, label, color, linestyle, linewidth)
        add_output_distortion_marker(ax, repair_rows, fraction=0.10)

        ax.set_title(f"Layer {layer_id}", fontsize=args.title_font_size, pad=8)
        ax.set_xlim(0.0, max_x)
        ax.set_ylim(0.0, 100.0)
        ax.set_xlabel("Selected block fraction (%)", fontsize=args.label_font_size)
        ax.tick_params(axis="both", labelsize=args.tick_font_size)
        ax.grid(True, alpha=0.22, linewidth=0.7)
        ax.legend(frameon=False, loc="upper left", fontsize=args.legend_font_size)
        if max_x <= 20.0:
            ax.set_xticks(np.arange(0.0, max_x + 1e-9, 5.0))
        else:
            ax.set_xticks(np.arange(0.0, max_x + 1e-9, 10.0))
        ax.set_yticks(np.arange(0.0, 101.0, 20.0))

    axes[0].set_ylabel("Layer Output Reduction (%)", fontsize=args.label_font_size)

    output_dir = args.output_dir or (args.summary_json.parent / "plots" / "observation23")
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = ["png", "pdf"] if args.format == "both" else [args.format]

    saved_paths: list[Path] = []
    filename_stem = make_output_filename_stem(args.module_name, args.layers)
    for fmt in formats:
        output_path = output_dir / f"{filename_stem}.{fmt}"
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        saved_paths.append(output_path)
    plt.close(fig)
    return saved_paths


def main() -> int:
    args = parse_args()
    rows_by_layer = load_layer_repair_rows(args.summary_json, args.layers)
    saved_paths = save_two_layer_plot(args, rows_by_layer)
    for path in saved_paths:
        print(f"[done] wrote: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
