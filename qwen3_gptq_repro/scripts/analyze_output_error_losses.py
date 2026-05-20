#!/usr/bin/env python3
"""Export SmoothBlock search losses to a two-sheet XLSX report.

输入:
  读取 qwen3_smooth_block_mixed.py 生成的 metadata JSON, 通常叫
  qwen3_smooth_block_mixed_metadata.json。脚本会从 metadata 中的
  results_summary.search 字段提取每个模块的:
    - loss
    - loss_before_residual
    - selected_blocks / total_blocks
  然后按 layer_id 聚合。

主要参数:
  --metadata:
    必填。输入的 qwen3_smooth_block_mixed_metadata.json 路径。
  --output:
    可选。输出 XLSX 文件路径。默认写到 metadata 同目录下的
    output_error_loss_analysis.xlsx。
  --sort-by:
    可选。控制每个 sheet 内 layer 排序依据, 支持 total / mean / max,
    默认 total。
  --csv-dir:
    可选。如果指定, 除 XLSX 外还会额外导出每个 metric 一个 CSV 文件。

运行示例:
  cd /root/autodl-tmp/Zip/qwen3_gptq_repro
  python scripts/analyze_output_error_losses.py \
    --metadata output/smooth_v16_b32/qwen3_smooth_block_mixed_metadata.json \
    --output output/smooth_v16_b32/output_error_loss_analysis.xlsx \
    --sort-by total \
    --csv-dir output/smooth_v16_b32/output_error_loss_csv

输出:
  1. 一个 XLSX 报告, 默认包含两个 sheet:
       - loss
       - loss_before_residual
  2. 每个 sheet 一行代表一层, 列包括 rank/layer_id/total/mean/max/max_module,
     以及 q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj 等模块列。
  3. 如果指定 --csv-dir, 同时写出 loss.csv 和 loss_before_residual.csv。

说明:
  这个脚本不重新跑模型、不重新量化、不画图; 它只是把已有 metadata 中的
  SmoothBlock 搜索误差整理成便于查看的 Excel/CSV 报告。XLSX 由标准库
  zipfile 直接写出, 不依赖 openpyxl/xlsxwriter。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


METRICS = ("loss", "loss_before_residual")
DEFAULT_MODULE_ORDER = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
LAYER_RE = re.compile(r"model\.layers\.(?P<layer_id>\d+)\.(?P<suffix>.+)$")




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SmoothBlock metadata loss/loss_before_residual by layer id."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to qwen3_smooth_block_mixed_metadata.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output XLSX path. Default: <metadata_dir>/output_error_loss_analysis.xlsx.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["total", "mean", "max"],
        default="total",
        help="Aggregate column used to sort rows within each sheet.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Optional directory to also write one CSV per metric.",
    )
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def split_module_name(module_name: str) -> tuple[int | None, str, str]:
    match = LAYER_RE.match(module_name)
    if match is None:
        suffix = module_name
        layer_id = None
    else:
        suffix = match.group("suffix")
        layer_id = int(match.group("layer_id"))
    module_type = suffix.rsplit(".", 1)[-1]
    return layer_id, suffix, module_type


def collect_module_records(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    search = metadata.get("results_summary", {}).get("search", {})
    if not isinstance(search, dict):
        raise ValueError(
            "metadata missing results_summary.search dict; "
            f"detected schema: {describe_metadata_schema(metadata)}"
        )

    records: list[dict[str, Any]] = []
    for search_key, group_payload in search.items():
        if not isinstance(group_payload, dict):
            continue
        group_name, _, prefix_from_key = str(search_key).partition(":")
        best_alpha = _as_float(group_payload.get("best_alpha"))
        group_loss = _as_float(group_payload.get("loss"))
        modules = group_payload.get("modules", {})
        if not isinstance(modules, dict):
            continue

        for module_name, module_stats in modules.items():
            if not isinstance(module_stats, dict):
                continue
            layer_id, suffix, module_type = split_module_name(module_name)
            selected_blocks = module_stats.get("selected_blocks")
            total_blocks = module_stats.get("total_blocks")
            selected_ratio = None
            if isinstance(selected_blocks, int) and isinstance(total_blocks, int) and total_blocks > 0:
                selected_ratio = selected_blocks / total_blocks
            records.append(
                {
                    "module_name": module_name,
                    "layer_id": layer_id,
                    "module_suffix": suffix,
                    "module_type": module_type,
                    "group": group_name,
                    "prefix": prefix_from_key,
                    "best_alpha": best_alpha,
                    "group_loss": group_loss,
                    "loss": _as_float(module_stats.get("loss")),
                    "loss_before_residual": _as_float(module_stats.get("loss_before_residual")),
                    "selected_blocks": selected_blocks,
                    "total_blocks": total_blocks,
                    "selected_ratio": selected_ratio,
                }
            )
    return records


def module_order(records: list[dict[str, Any]]) -> list[str]:
    seen = {record["module_type"] for record in records}
    ordered = [name for name in DEFAULT_MODULE_ORDER if name in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def build_metric_rows(
    records: list[dict[str, Any]],
    metric: str,
    ordered_modules: list[str],
    sort_by: str,
) -> list[dict[str, Any]]:
    layers: dict[int | str, dict[str, Any]] = {}
    for record in records:
        layer_key: int | str = record["layer_id"] if record["layer_id"] is not None else "unknown"
        layer = layers.setdefault(
            layer_key,
            {
                "layer_id": layer_key,
                "values": {},
                "selected_blocks": 0,
                "total_blocks": 0,
            },
        )
        value = record.get(metric)
        if value is not None:
            layer["values"][record["module_type"]] = value
        if isinstance(record.get("selected_blocks"), int):
            layer["selected_blocks"] += record["selected_blocks"]
        if isinstance(record.get("total_blocks"), int):
            layer["total_blocks"] += record["total_blocks"]

    rows: list[dict[str, Any]] = []
    for layer in layers.values():
        values = layer["values"]
        finite_values = [values[name] for name in ordered_modules if values.get(name) is not None]
        total = sum(finite_values) if finite_values else None
        mean = total / len(finite_values) if total is not None and finite_values else None
        max_value = max(finite_values) if finite_values else None
        max_module = None
        if max_value is not None:
            for module_type in ordered_modules:
                if values.get(module_type) == max_value:
                    max_module = module_type
                    break
        total_blocks = layer["total_blocks"]
        selected_blocks = layer["selected_blocks"]
        row = {
            "rank": 0,
            "layer_id": layer["layer_id"],
            "total": total,
            "mean": mean,
            "max": max_value,
            "max_module": max_module,
            "selected_blocks": selected_blocks,
            "total_blocks": total_blocks,
            "selected_ratio": (selected_blocks / total_blocks) if total_blocks else None,
        }
        for module_type in ordered_modules:
            row[module_type] = values.get(module_type)
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row.get(sort_by) is not None,
            row.get(sort_by) if row.get(sort_by) is not None else float("-inf"),
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def column_headers(ordered_modules: list[str]) -> list[str]:
    return [
        "rank",
        "layer_id",
        "total",
        "mean",
        "max",
        "max_module",
        *ordered_modules,
        "selected_blocks",
        "total_blocks",
        "selected_ratio",
    ]


def _column_letter(index: int) -> str:
    letters = []
    while index:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def _cell_xml(row_idx: int, col_idx: int, value: Any) -> str:
    ref = f"{_column_letter(col_idx)}{row_idx}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return f'<c r="{ref}"><v>{numeric:.17g}</v></c>'
        return f'<c r="{ref}"/>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _sheet_xml(headers: list[str], rows: list[dict[str, Any]]) -> str:
    xml_rows = []
    header_cells = "".join(_cell_xml(1, col_idx, header) for col_idx, header in enumerate(headers, start=1))
    xml_rows.append(f'<row r="1">{header_cells}</row>')
    for row_idx, row in enumerate(rows, start=2):
        cells = "".join(_cell_xml(row_idx, col_idx, row.get(header)) for col_idx, header in enumerate(headers, start=1))
        xml_rows.append(f'<row r="{row_idx}">{cells}</row>')
    max_ref = f"{_column_letter(len(headers))}{len(rows) + 1}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="A1:{max_ref}"/>'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '<selection pane="bottomLeft"/></sheetView></sheetViews>'
        '<sheetData>'
        + "".join(xml_rows)
        + '</sheetData>'
        '<autoFilter ref="A1:'
        + max_ref
        + '"/>'
        '</worksheet>'
    )


def write_xlsx(path: Path, sheets: dict[str, tuple[list[str], list[dict[str, Any]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = list(sheets.keys())

    workbook_sheets = []
    workbook_rels = []
    content_overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for sheet_id, name in enumerate(sheet_names, start=1):
        safe_name = escape(name[:31])
        workbook_sheets.append(f'<sheet name="{safe_name}" sheetId="{sheet_id}" r:id="rId{sheet_id}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{sheet_id}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{sheet_id}.xml"/>'
        )
        content_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{sheet_id}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    styles_rid = len(sheet_names) + 1
    workbook_rels.append(
        f'<Relationship Id="rId{styles_rid}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(content_overrides)
        + '</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>'
        + "".join(workbook_sheets)
        + '</sheets>'
        '</workbook>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(workbook_rels)
        + '</Relationships>'
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '</styleSheet>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/styles.xml", styles_xml)
        for sheet_id, name in enumerate(sheet_names, start=1):
            headers, rows = sheets[name]
            zf.writestr(f"xl/worksheets/sheet{sheet_id}.xml", _sheet_xml(headers, rows))


def write_metric_csvs(csv_dir: Path, sheets: dict[str, tuple[list[str], list[dict[str, Any]]]]) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    for metric, (headers, rows) in sheets.items():
        path = csv_dir / f"{metric}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    args = parse_args()
    metadata_path = args.metadata.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else metadata_path.parent / "output_error_loss_analysis.xlsx"
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = collect_module_records(metadata)
    if not records:
        raise RuntimeError(f"no module loss records found in {metadata_path}")

    ordered_modules = module_order(records)
    headers = column_headers(ordered_modules)
    sheets = {
        metric: (headers, build_metric_rows(records, metric, ordered_modules, args.sort_by))
        for metric in METRICS
    }
    write_xlsx(output_path, sheets)
    if args.csv_dir is not None:
        write_metric_csvs(args.csv_dir.resolve(), sheets)

    print(f"metadata: {metadata_path}")
    print(f"records : {len(records)} modules")
    print(f"output  : {output_path}")
    if args.csv_dir is not None:
        print(f"csv-dir : {args.csv_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
