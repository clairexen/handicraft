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
        "--list",
        action="store_true",
        help="List summary statistics for each source (default action)",
    )
    return parser.parse_args()


def load_history(pt_path: pathlib.Path) -> Tuple[str, List[Dict[str, float]]]:
    import torch

    payload = torch.load(pt_path, map_location="cpu")
    history = payload.get("loss_history", []) if isinstance(payload, dict) else []
    normalized = [normalize_entry(item) for item in history if isinstance(item, dict)]
    return pt_path.name, normalized


def normalize_entry(entry: Dict[str, float]) -> Dict[str, float]:
    out = dict(entry)

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


def main() -> None:
    args = parse_args()
    sources: List[Tuple[str, List[Dict[str, float]]]] = []
    for pt_path in args.pt:
        if not pt_path.exists():
            print(f"warning: missing checkpoint {pt_path}")
            continue
        label, history = load_history(pt_path)
        sources.append((label, history))
    actions = [args.list] if args.list else []
    if not actions:
        actions.append(True)  # default to list
    if actions[0]:
        for label, history in sources:
            summarize_source(label, history)


if __name__ == "__main__":
    main()
