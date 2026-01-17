#!/usr/bin/env python3
"""Inspect loss history data from checkpoint files."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Dict, Iterable, List, Tuple

LEGACY_TARGET_FIELDS = {
    "train_loss_learned": "train_target",
    "train_loss_nogrce_learned": "train_target_nogrce",
    "test_loss_learned": "test_target",
    "test_loss_nogrce_learned": "test_target_nogrce",
}

LEGACY_PLAIN_FIELDS = {
    "train_loss_nothink": "train_loss_plain",
    "test_loss_nothink": "test_loss_plain",
    "train_loss_nothink_learned": "train_target",
    "test_loss_nothink_learned": "test_target",
}

ALLOWED_FIELDS = {
    "step",
    "train_loss",
    "train_target",
    "train_loss_nogrce",
    "train_target_nogrce",
    "test_loss",
    "test_target",
    "test_loss_nogrce",
    "test_target_nogrce",
    "train_loss_plain",
    "test_loss_plain",
    "train_wall_seconds",
    "unix_time",
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
        "--list",
        action="store_true",
        help="List summary statistics for each source (default action)",
    )
    parser.add_argument(
        "--write-json",
        type=pathlib.Path,
        help="Write normalized records to JSON file",
    )
    return parser.parse_args()


def load_history(pt_path: pathlib.Path) -> Tuple[str, List[Dict[str, float]]]:
    import torch

    payload = torch.load(pt_path, map_location="cpu")
    history = payload.get("loss_history", []) if isinstance(payload, dict) else []
    normalized = [normalize_entry(item) for item in history if isinstance(item, dict)]
    return pt_path.name, normalized


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
    if "step" in out:
        try:
            out["step"] = int(out["step"])
        except (TypeError, ValueError):
            out.pop("step", None)

    def apply_alias(mapping: Dict[str, str]) -> None:
        for legacy_key, new_key in mapping.items():
            if legacy_key not in out:
                continue
            value = out.pop(legacy_key)
            if new_key not in out:
                out[new_key] = value

    apply_alias(LEGACY_TARGET_FIELDS)
    apply_alias(LEGACY_PLAIN_FIELDS)
    filtered = {k: v for k, v in out.items() if k in ALLOWED_FIELDS}
    return filtered


def summarize_source(label: str, records: List[Dict[str, float]]) -> None:
    print(f"Source: {label} ({len(records)} records)")
    if not records:
        return
    fields = sorted({key for rec in records for key in rec if key in ALLOWED_FIELDS})
    for field in fields:
        values = [float(rec[field]) for rec in records if field in rec]
        if not values:
            continue
        count = len(values)
        min_val = min(values)
        max_val = max(values)
        mean = sum(values) / count
        variance = sum((val - mean) ** 2 for val in values) / count
        stddev = math.sqrt(variance)
        print(
            f"  {field}: count={count} min={min_val:.4f} max={max_val:.4f} "
            f"mean={mean:.4f} std={stddev:.4f}"
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


def main() -> None:
    args = parse_args()
    sources: List[Tuple[str, List[Dict[str, float]]]] = []
    for pt_path in args.pt:
        if not pt_path.exists():
            print(f"warning: missing checkpoint {pt_path}")
            continue
        label, history = load_history(pt_path)
        sources.append((label, history))
    for json_path in args.json:
        if not json_path.exists():
            print(f"warning: missing json source {json_path}")
            continue
        label, history = load_json_history(json_path)
        sources.append((label, history))
    if not sources:
        print("no data sources provided")
        return
    performed = False
    if args.list or not args.write_json:
        for label, history in sources:
            summarize_source(label, history)
        performed = True
    if args.write_json:
        if len(sources) != 1:
            raise SystemExit("--write-json expects exactly one source (--pt or --json)")
        columns, matrix = combine_records(sources)
        json_text = format_table_json(columns, matrix)
        args.write_json.write_text(json_text + "\n")
        print(f"wrote {len(matrix)} rows to {args.write_json}")
        performed = True
    if not performed:
        print("no action taken (no list or write-json requested)")


if __name__ == "__main__":
    main()
