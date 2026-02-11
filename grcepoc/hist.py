#!/usr/bin/env python3
"""Inspect loss history data from checkpoint files."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

LEGACY_SPECIAL_FIELDS = {
    # "train_loss_noatt": "train_loss_noattn",
    # "test_loss_noatt": "test_loss_noattn",
}

ALLOWED_FIELDS = {
    "step",
    "step_split_eval",
    "batch_layout",
    "train_loss",
    # "train_loss_target",
    "train_loss_encode",
    "train_loss_decode",
    "train_loss_forward",
    "train_loss_noattn",
    "test_loss",
    # "test_loss_target",
    "test_loss_encode",
    "test_loss_decode",
    "test_loss_forward",
    "test_loss_noattn",
    "train_wall_seconds",
    "unix_time",
    "corpus",
    "train_cursor",
    "test_cursor",
    "train_cycle",
    "test_cycle",
    "learning_rate"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pt",
        action="append",
        type=pathlib.Path,
        default=[],
        help="Checkpoint .pt file to inspect (repeatable)",
    )
    parser.add_argument(
        "--json",
        action="append",
        type=pathlib.Path,
        default=[],
        help="Pre-exported history JSON file (see --write-json)",
    )
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=pathlib.Path("model"),
        help="Directory to search for checkpoints when --pt/--json are omitted",
    )
    parser.add_argument(
        "--list",
        nargs="*",
        help="List summary stats (optionally specify metric names)",
    )
    parser.add_argument(
        "--write-json",
        type=pathlib.Path,
        help="Write normalized records to JSON file",
    )
    parser.add_argument(
        "--write-json-dir",
        type=pathlib.Path,
        help="Write per-source JSON files into the specified directory",
    )
    parser.add_argument(
        "--step-period",
        type=int,
        default=0,
        help="If >0, reshape step traces into segments of this many samples and overlay them",
    )
    parser.add_argument(
        "--plot-relative",
        action="store_true",
        help="Subtract the first value of each trace/segment before plotting",
    )
    parser.add_argument(
        "--plot-deltas",
        action="store_true",
        help="Plot first differences instead of raw values",
    )
    parser.add_argument(
        "--plot-sum",
        action="store_true",
        help="Plot the cumulative sum (integral) of each trace",
    )
    parser.add_argument(
        "--filter",
        type=int,
        default=0,
        help="For each group of N samples, drop the smallest/largest 25% before plotting",
    )
    parser.add_argument(
        "--median",
        type=int,
        default=0,
        help="For each group of N samples, replace the block with its median",
    )
    parser.add_argument(
        "--plot-steps",
        nargs="*",
        help="Plot step vs metric (default: test_loss); use ':' to split metrics into subplots",
    )
    parser.add_argument(
        "--plot-time",
        nargs="*",
        help="Plot train_wall_seconds vs metric (default: test_loss); ':' makes separate subplots",
    )
    parser.add_argument(
        "--plot-timestamp",
        nargs="*",
        help="Plot unix_time vs metric (default: test_loss); ':' makes separate subplots",
    )
    parser.add_argument(
        "--skip-split-evals",
        action="store_true",
        help="Ignore records where step_split_eval == 1.0 (pure split evaluations)",
    )
    parser.add_argument(
        "--fit-line",
        type=int,
        default=0,
        help="Least-squares fit of a line to the last N samples of each plotted trace",
    )
    parser.add_argument(
        "--fit-quad",
        type=int,
        default=0,
        help="Least-squares fit of a quadratic to the last N samples of each plotted trace",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Plot raw values as scatter points instead of continuous lines",
    )
    parser.add_argument(
        "--backtrace",
        type=int,
        default=0,
        help="Ignore the last N samples from each history before plotting/exporting",
    )
    return parser.parse_args()


def load_history(pt_path: pathlib.Path) -> Tuple[str, List[Dict[str, float]], bool]:
    import torch

    payload = torch.load(pt_path, map_location="cpu")
    if not isinstance(payload, dict):
        return pt_path.name, [], False
    history_payload = payload.get("loss_history")
    if not isinstance(history_payload, list):
        return pt_path.name, [], False
    normalized = [
        normalize_entry(item) for item in history_payload if isinstance(item, dict)
    ]
    return pt_path.name, normalized, True


def load_json_history(json_path: pathlib.Path) -> Tuple[str, List[Dict[str, float]]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    columns = payload.get("columns", [])
    rows = payload.get("data", [])
    records: List[Dict[str, float]] = []
    for row in rows:
        record = {}
        for idx, field in enumerate(columns):
            if idx < len(row):
                value = row[idx]
                if value is not None:
                    record[field] = value
        records.append(normalize_entry(record))
    return json_path.name, records


def normalize_entry(entry: Dict[str, float]) -> Dict[str, float]:
    out = dict(entry)
    for key in ("step", "train_cursor", "test_cursor"):
        if key in out:
            try:
                out[key] = int(out[key])
            except (TypeError, ValueError):
                out.pop(key, None)

    def apply_alias(mapping: Dict[str, str]) -> None:
        for legacy_key, new_key in mapping.items():
            if legacy_key not in out:
                continue
            value = out.pop(legacy_key)
            if new_key not in out:
                out[new_key] = value

    apply_alias(LEGACY_SPECIAL_FIELDS)
    filtered = {k: v for k, v in out.items() if k in ALLOWED_FIELDS}
    return filtered


def _skip_split_eval(record: Dict[str, float]) -> bool:
    value = record.get("step_split_eval")
    if value is None:
        return False
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def summarize_source(label: str, records: List[Dict[str, float]], filters: List[str] | None = None) -> None:
    print(f"\nSource: {label} ({len(records)} records)")
    if not records:
        return
    fields = sorted({key for rec in records for key in rec if key in ALLOWED_FIELDS})
    if filters:
        wanted = set(filters)
        fields = [field for field in fields if field in wanted]
    for field in fields:
        raw_values = [rec[field] for rec in records if field in rec]
        if not raw_values:
            continue
        numeric_values: List[float] = []
        non_numeric_samples: List[str] = []
        for value in raw_values:
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
            else:
                try:
                    numeric_values.append(float(value))
                except (TypeError, ValueError):
                    non_numeric_samples.append(str(value))
        if numeric_values:
            count = len(numeric_values)
            min_val = min(numeric_values)
            max_val = max(numeric_values)
            mean = sum(numeric_values) / count
            variance = sum((val - mean) ** 2 for val in numeric_values) / count
            stddev = math.sqrt(variance)
            print(
                f"  {field}: count={count} min={min_val:.4f} max={max_val:.4f} "
                f"mean={mean:.4f} std={stddev:.4f}"
            )
            continue
        if non_numeric_samples:
            uniq = sorted(set(non_numeric_samples))
            preview = ", ".join(uniq[:10])
            suffix = "..." if len(uniq) > 10 else ""
            print(
                f"  {field}: {len(uniq)} unique values ({preview}{suffix})"
            )


def combine_records(sources: List[Tuple[str, List[Dict[str, float]]]]) -> Tuple[List[str], List[List[float]]]:
    field_set = set()
    for _, history in sources:
        for record in history:
            field_set.update(record.keys())
    fields = sorted(field_set)
    data: List[List[object]] = []
    for _, history in sources:
        for record in history:
            row = [record.get(field) if field in record else None for field in fields]
            data.append(row)
    return fields, data


def format_table_json(columns: List[str], rows: List[List[float]]) -> str:
    def encode_scalar(value: float | None) -> str:
        if value is None:
            return "null"
        if isinstance(value, float):
            return f"{value:.5f}"
        return json.dumps(value)

    column_line = "[" + ",".join(json.dumps(col) for col in columns) + "]"
    if rows:
        row_lines = [
            "    [" + ",".join(encode_scalar(val) for val in row) + "]"
            for row in rows
        ]
        data_block = "[\n" + ",\n".join(row_lines) + "\n  ]"
    else:
        data_block = "[]"
    lines = [
        "{",
        f"  \"columns\": {column_line},",
        f"  \"data\": {data_block}",
        "}",
    ]
    return "\n".join(lines)


def _series_from_field(history: List[Dict[str, float]], field: str, default_sequence=False) -> List[float]:
    values: List[float] = []
    for idx, rec in enumerate(history):
        if default_sequence and field == "step" and "step" not in rec:
            values.append(float(idx + 1))
            continue
        value = rec.get(field)
        if value is None:
            values.append(float("nan"))
        else:
            values.append(float(value))
    return values


def _split_segments(values: List[float], period: int) -> List[List[float]]:
    if period <= 1 or period > len(values):
        return [values]
    segments: List[List[float]] = []
    chunk_count = len(values) // period
    for idx in range(chunk_count):
        start = idx * period
        segments.append(values[start : start + period])
    return segments


def _apply_transforms(
    x_values: List[float],
    y_values: List[float],
    *,
    relative: bool,
    deltas: bool,
    cumulative: bool,
) -> Tuple[List[float], List[float]]:
    if not y_values:
        return x_values, y_values
    y_proc = list(y_values)
    x_proc = list(x_values)
    if relative:
        base = y_proc[0]
        y_proc = [val - base for val in y_proc]
    if deltas and len(y_proc) > 1:
        y_proc = [y_proc[i + 1] - y_proc[i] for i in range(len(y_proc) - 1)]
        x_proc = x_proc[1:]
    if cumulative:
        total = 0.0
        cumulative_vals = []
        for val in y_proc:
            total += val
            cumulative_vals.append(total)
        y_proc = cumulative_vals
    return x_proc, y_proc


def _apply_filter_groups(
    x_values: List[float],
    y_values: List[float],
    group_size: int,
) -> Tuple[List[float], List[float]]:
    if group_size <= 1 or not y_values:
        return x_values, y_values
    filtered_x: List[float] = []
    filtered_y: List[float] = []
    total = len(y_values)
    for start in range(0, total, group_size):
        end = min(total, start + group_size)
        block_x = x_values[start:end]
        block_y = y_values[start:end]
        if not block_y:
            continue
        if any(math.isnan(val) for val in block_y):
            filtered_x.extend(block_x)
            filtered_y.extend(block_y)
            continue
        drop = int(len(block_y) * 0.25)
        if drop <= 0 or drop * 2 >= len(block_y):
            mask = [True] * len(block_y)
        else:
            sorted_indices = sorted(range(len(block_y)), key=lambda idx: block_y[idx])
            drop_set = set(sorted_indices[:drop] + sorted_indices[-drop:])
            mask = [idx not in drop_set for idx in range(len(block_y))]
        for keep, x_val, y_val in zip(mask, block_x, block_y):
            if keep:
                filtered_x.append(x_val)
                filtered_y.append(y_val)
    return filtered_x, filtered_y


def _apply_median_groups(
    x_values: List[float],
    y_values: List[float],
    group_size: int,
) -> Tuple[List[float], List[float]]:
    if group_size <= 1 or not y_values:
        return x_values, y_values
    result_x: List[float] = []
    result_y: List[float] = []
    total = len(y_values)
    for start in range(0, total, group_size):
        end = min(total, start + group_size)
        block_y = y_values[start:end]
        block_x = x_values[start:end]
        if not block_y:
            continue
        if len(block_y) == 1:
            result_x.append(block_x[0])
            result_y.append(block_y[0])
            continue
        clean = [val for val in block_y if not math.isnan(val)]
        if not clean:
            continue
        median_val = float(np.median(clean))
        result_x.append(block_x[-1])
        result_y.append(median_val)
    return result_x, result_y


def _fit_line(points: List[Tuple[float, float]]) -> Tuple[float, float] | None:
    if len(points) < 2:
        return None
    mean_x = sum(pt[0] for pt in points) / len(points)
    mean_y = sum(pt[1] for pt in points) / len(points)
    denom = sum((pt[0] - mean_x) ** 2 for pt in points)
    if denom == 0:
        return None
    numer = sum((pt[0] - mean_x) * (pt[1] - mean_y) for pt in points)
    slope = numer / denom
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _fit_quadratic(points: List[Tuple[float, float]]) -> Tuple[float, float, float] | None:
    if len(points) < 3:
        return None
    xs = np.array([pt[0] for pt in points], dtype=float)
    ys = np.array([pt[1] for pt in points], dtype=float)
    if np.all(xs == xs[0]):
        return None
    design = np.vstack([xs**2, xs, np.ones_like(xs)]).T
    coeffs, *_ = np.linalg.lstsq(design, ys, rcond=None)
    a, b, c = coeffs.tolist()
    return a, b, c


def _build_metric_groups(values: list[str] | None, fallback: list[str]) -> list[list[str]]:
    if not values:
        return [list(fallback)]
    groups: list[list[str]] = [[]]
    for token in values:
        if token == ":":
            if groups[-1]:
                groups.append([])
            continue
        groups[-1].append(token)
    groups = [group for group in groups if group]
    return groups or [list(fallback)]


def plot_metric_traces(
    sources: List[Tuple[str, List[Dict[str, float]]]],
    metric_groups: List[List[str]],
    x_field: str,
    x_label: str,
    *,
    step_period: int = 0,
    relative: bool = False,
    deltas: bool = False,
    cumulative: bool = False,
    fit_line: int = 0,
    fit_quad: int = 0,
    scatter: bool = False,
    value_filter: int = 0,
    group_median: int = 0,
) -> None:
    num_groups = max(1, len(metric_groups))
    fig, axes = plt.subplots(
        num_groups,
        1,
        figsize=(10, 4 * num_groups),
        sharex=True,
    )
    if num_groups == 1:
        axes = [axes]
    for idx_ax, (ax, metrics) in enumerate(zip(axes, metric_groups)):
        for label, history in sources:
            if not history:
                continue
            x_values = _series_from_field(history, x_field, default_sequence=True)
            if x_field == "train_wall_seconds":
                x_values = [val / 3600.0 if not math.isnan(val) else val for val in x_values]
            if not any(not math.isnan(val) for val in x_values):
                continue
            for metric in metrics:
                y_values = _series_from_field(history, metric)
                if not any(not math.isnan(val) for val in y_values):
                    continue
                x_series = x_values
                y_series = y_values
                if value_filter > 1:
                    x_series, y_series = _apply_filter_groups(x_series, y_series, value_filter)
                if group_median > 1:
                    x_series, y_series = _apply_median_groups(x_series, y_series, group_median)
                if x_field == "step" and step_period > 1:
                    y_segments = _split_segments(y_series, step_period)
                    x_segments = [list(range(len(seg))) for seg in y_segments]
                else:
                    y_segments = [y_series]
                    x_segments = [x_series]
                for x_seg, y_seg in zip(x_segments, y_segments):
                    x_plot, y_plot = _apply_transforms(
                        x_seg,
                        y_seg,
                        relative=relative,
                        deltas=deltas,
                        cumulative=cumulative,
                    )
                    if not y_plot:
                        continue
                    label_name = f"{label} – {metric}"
                    if scatter:
                        ax.scatter(
                            x_plot,
                            y_plot,
                            label=label_name,
                            s=9,
                            alpha=0.9,
                        )
                    else:
                        ax.plot(
                            x_plot,
                            y_plot,
                            label=label_name,
                            linewidth=2,
                        )
                    if (fit_line or fit_quad) and len(x_plot) >= 2:
                        pairs = [
                            (x_val, y_val)
                            for x_val, y_val in zip(x_plot, y_plot)
                            if not math.isnan(x_val) and not math.isnan(y_val)
                        ]
                        if pairs:
                            max_tail = max(fit_line, fit_quad)
                            tail = pairs[-min(max_tail if max_tail > 0 else len(pairs), len(pairs)) :]
                            if fit_line:
                                line_tail = tail[-min(fit_line, len(tail)) :]
                                line = _fit_line(line_tail)
                                if line is not None:
                                    slope, intercept = line
                                    x_start = line_tail[0][0]
                                    x_end = line_tail[-1][0]
                                    if x_start != x_end:
                                        y_start = slope * x_start + intercept
                                        y_end = slope * x_end + intercept
                                        delta_y = y_end - y_start
                                        print(
                                            f"fit-line: {label_name} slope={slope:.4g} delta_x={x_end - x_start:.4g} delta_y={delta_y:.4g}"
                                        )
                                        ax.plot(
                                            [x_start, x_end],
                                            [y_start, y_end],
                                            linestyle="--",
                                            alpha=0.7,
                                            linewidth=2,
                                            label=f"{label} – {metric} fit",
                                        )
                                        x_extra = x_end + (x_end - x_start)
                                        y_extra = slope * x_extra + intercept
                                        ax.plot(
                                            [x_end, x_extra],
                                            [y_end, y_extra],
                                            linestyle="--",
                                            alpha=0.5,
                                            linewidth=2,
                                            label=f"{label} – {metric} fit extrap",
                                        )
                            if fit_quad:
                                quad_tail = tail[-min(fit_quad, len(tail)) :]
                                quad = _fit_quadratic(quad_tail)
                                if quad is not None:
                                    a, b, c = quad
                                    xs = [pt[0] for pt in quad_tail]
                                    x_start = xs[0]
                                    x_end = xs[-1]
                                    sample_x = [x_start + (x_end - x_start) * t / 20 for t in range(21)]
                                    y_vals = [a * x ** 2 + b * x + c for x in sample_x]
                                    print(
                                        f"fit-quad: {label_name} a={a:.4g} b={b:.4g} c={c:.4g}"
                                    )
                                    ax.plot(
                                        sample_x,
                                        y_vals,
                                        linestyle=":",
                                        alpha=0.7,
                                        linewidth=2,
                                        label=f"{label} – {metric} quad",
                                    )
                                    x_extra = x_end + (x_end - x_start)
                                    sample_extra = np.linspace(x_end, x_extra, 20)
                                    y_extra = a * sample_extra**2 + b * sample_extra + c
                                    ax.plot(
                                        sample_extra,
                                        y_extra,
                                        linestyle=":",
                                        alpha=0.4,
                                        linewidth=2,
                                        label=f"{label} – {metric} quad extrap",
                                    )
        ylabel = ", ".join(metrics) if metrics else "metric"
        ax.set_ylabel(ylabel)
        if idx_ax == num_groups - 1:
            ax.set_xlabel(x_label)
        else:
            ax.set_xlabel("")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    auto_pt_sources: set[pathlib.Path] = set()
    source_path_lookup: Dict[str, pathlib.Path] = {}
    if not args.pt and not args.json:
        model_dir = args.model
        if model_dir.exists():
            pt_files = sorted(model_dir.glob("*.pt"))
            if pt_files:
                args.pt.extend(pt_files)
                auto_pt_sources = set(pt_files)
            else:
                json_files = sorted(model_dir.glob("*.json"))
                args.json.extend(json_files)
        else:
            print(f"warning: model directory {model_dir} not found")
    sources: List[Tuple[str, List[Dict[str, float]]]] = []
    for pt_path in args.pt:
        if not pt_path.exists():
            print(f"warning: missing checkpoint {pt_path}")
            continue
        label, history, has_history = load_history(pt_path)
        source_path_lookup[label] = pt_path
        if pt_path in auto_pt_sources and not has_history:
            print(f"Ignored: {pt_path.name}")
            continue
        sources.append((label, history))
    for json_path in args.json:
        if not json_path.exists():
            print(f"warning: missing json source {json_path}")
            continue
        label, history = load_json_history(json_path)
        source_path_lookup[label] = json_path
        sources.append((label, history))
    if args.backtrace > 0:
        trimmed_sources: List[Tuple[str, List[Dict[str, float]]]] = []
        for label, history in sources:
            if len(history) > args.backtrace:
                trimmed_sources.append((label, history[:-args.backtrace]))
            else:
                trimmed_sources.append((label, []))
        sources = trimmed_sources
    if args.skip_split_evals and sources:
        filtered_sources: List[Tuple[str, List[Dict[str, float]]]] = []
        for label, history in sources:
            filtered_history = [rec for rec in history if not _skip_split_eval(rec)]
            filtered_sources.append((label, filtered_history))
        sources = filtered_sources
    if not sources:
        print("no data sources provided")
        return
    performed = False
    list_filters = None
    if args.list is not None:
        list_filters = args.list if args.list else None
    if args.list is not None or not args.write_json:
        for label, history in sources:
            summarize_source(label, history, filters=list_filters)
        performed = True
    if args.write_json:
        if len(sources) != 1:
            raise SystemExit("--write-json expects exactly one source (--pt or --json)")
        columns, matrix = combine_records(sources)
        json_text = format_table_json(columns, matrix)
        args.write_json.write_text(json_text + "\n")
        print(f"wrote {len(matrix)} rows to {args.write_json}")
        performed = True
    if args.write_json_dir:
        args.write_json_dir.mkdir(parents=True, exist_ok=True)
        for label, history in sources:
            columns, matrix = combine_records([(label, history)])
            json_text = format_table_json(columns, matrix)
            safe_name = pathlib.Path(label).name
            output_name = pathlib.Path(safe_name).stem + ".json"
            output_path = args.write_json_dir / output_name
            output_path.write_text(json_text + "\n")
            print(f"wrote {len(matrix)} rows to {output_path}")
            src = source_path_lookup.get(label)
            if src is not None:
                log_candidate = src.with_suffix(".log")
                if log_candidate.exists():
                    dest_log = args.write_json_dir / log_candidate.name
                    shutil.copy(log_candidate, dest_log)
                    print(f"copied log to {dest_log}")
        performed = True
    if args.plot_steps is not None:
        metric_groups = _build_metric_groups(args.plot_steps, ["test_loss"])
        plot_metric_traces(
            sources,
            metric_groups,
            "step",
            "Step",
            step_period=args.step_period,
            relative=args.plot_relative,
            deltas=args.plot_deltas,
            cumulative=args.plot_sum,
            fit_line=args.fit_line,
            fit_quad=args.fit_quad,
            scatter=args.scatter,
            value_filter=args.filter,
            group_median=args.median,
        )
        performed = True
    if args.plot_time is not None:
        metric_groups = _build_metric_groups(args.plot_time, ["test_loss"])
        plot_metric_traces(
            sources,
            metric_groups,
            "train_wall_seconds",
            "Train wall hours",
            relative=args.plot_relative,
            deltas=args.plot_deltas,
            cumulative=args.plot_sum,
            fit_line=args.fit_line,
            fit_quad=args.fit_quad,
            scatter=args.scatter,
            value_filter=args.filter,
            group_median=args.median,
        )
        performed = True
    if args.plot_timestamp is not None:
        metric_groups = _build_metric_groups(args.plot_timestamp, ["test_loss"])
        plot_metric_traces(
            sources,
            metric_groups,
            "unix_time",
            "Unix time",
            relative=args.plot_relative,
            deltas=args.plot_deltas,
            cumulative=args.plot_sum,
            fit_line=args.fit_line,
            fit_quad=args.fit_quad,
            scatter=args.scatter,
            value_filter=args.filter,
            group_median=args.median,
        )
        performed = True
    if not performed:
        print("no action taken (no list, plot or write-json requested)")


if __name__ == "__main__":
    main()
