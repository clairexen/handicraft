"""Plot train/test losses over steps for checkpoints."""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import json
import re

import matplotlib.pyplot as plt


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


def normalize_history_entry(entry: Dict[str, float]) -> Dict[str, float]:
    normalized = dict(entry)

    def apply_aliases(mapping: Dict[str, str]) -> None:
        for legacy_key, new_key in mapping.items():
            if legacy_key not in normalized:
                continue
            value = normalized.pop(legacy_key)
            if new_key not in normalized:
                normalized[new_key] = value

    apply_aliases(LEGACY_TARGET_FIELDS)
    apply_aliases(LEGACY_PLAIN_FIELDS)
    return normalized


@dataclass
class LossRecord:
    model_path: pathlib.Path
    steps: List[int]
    train: List[float]
    test: List[float]
    train_nogrce: Optional[List[float]] = None
    test_nogrce: Optional[List[float]] = None


def load_records(json_paths: Iterable[pathlib.Path]) -> List[LossRecord]:
    records: List[LossRecord] = []
    for json_path in json_paths:
        if not json_path.exists():
            print(f"warning: {json_path} not found, skipping")
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        columns = payload.get("columns", [])
        data = payload.get("data", [])
        column_index = {name: idx for idx, name in enumerate(columns)}

        def extract_series(field: str, *, as_int: bool = False) -> Optional[List[float | int]]:
            idx = column_index.get(field)
            if idx is None:
                return None
            series: List[float | int] = []
            for row in data:
                value = row[idx] if idx < len(row) else None
                if value is None:
                    series.append(float("nan"))
                else:
                    series.append(int(value) if as_int else float(value))
            return series

        steps_series = extract_series("step", as_int=True)
        if steps_series is None:
            steps = list(range(1, len(data) + 1))
        else:
            steps = [int(val) for val in steps_series]
        train = extract_series("train_loss") or []
        test = extract_series("test_loss") or []
        train_ng = extract_series("train_loss_nogrce")
        test_ng = extract_series("test_loss_nogrce")
        if not train and not test:
            continue
        records.append(
            LossRecord(
                model_path=json_path,
                steps=steps,
                train=[float(v) for v in train],
                test=[float(v) for v in test],
                train_nogrce=[float(v) for v in train_ng] if train_ng else None,
                test_nogrce=[float(v) for v in test_ng] if test_ng else None,
            )
        )
    return records


def load_store(path: pathlib.Path) -> List[LossRecord]:
    if not path.exists():
        print(f"warning: stored file {path} not found")
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload if isinstance(payload, list) else [payload]
    records: List[LossRecord] = []
    for entry in entries:
        records.append(
            LossRecord(
                model_path=pathlib.Path(entry.get("model", path.name)),
                steps=list(map(int, entry.get("steps", []))),
                train=list(map(float, entry.get("train", [])))
                if entry.get("train")
                else [],
                test=list(map(float, entry.get("test", [])))
                if entry.get("test")
                else [],
                train_nogrce=(
                    list(map(float, entry.get("train_nogrce", [])))
                    if entry.get("train_nogrce")
                    else None
                ),
                test_nogrce=(
                    list(map(float, entry.get("test_nogrce", [])))
                    if entry.get("test_nogrce")
                    else None
                ),
            )
        )
    return records


def store_records(
    records: List[LossRecord],
    out_path: pathlib.Path,
    *,
    include_train: bool,
    include_test: bool,
    include_nogrce: bool,
) -> None:
    payload = [
        {
            "model": rec.model_path.name,
            "steps": rec.steps,
            **({"train": rec.train} if include_train else {}),
            **({"test": rec.test} if include_test else {}),
            **(
                {"train_nogrce": rec.train_nogrce}
                if include_train and include_nogrce and rec.train_nogrce is not None
                else {}
            ),
            **(
                {"test_nogrce": rec.test_nogrce}
                if include_test and include_nogrce and rec.test_nogrce is not None
                else {}
            ),
        }
        for rec in records
    ]
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"stored {len(records)} record(s) to {out_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=pathlib.Path,
        help=".pt checkpoint files (defaults to model/*.pt if omitted)",
    )
    parser.add_argument("--train", action="store_true", help="plot train losses only")
    parser.add_argument(
        "--train-and-test",
        action="store_true",
        help="plot both train and test losses (default is test only)",
    )
    parser.add_argument("--store", type=pathlib.Path, help="write loss curves to JSON file")
    parser.add_argument(
        "--stored",
        type=pathlib.Path,
        help="load JSON file produced by --store (skips .pt loading)",
    )
    parser.add_argument(
        "--no-nogrce",
        action="store_true",
        help="suppress GRCE-disabled (nogrce) traces",
    )
    parser.add_argument(
        "--default-test",
        action="store_true",
        help="When set, default view is test-only with nogrce traces hidden",
    )
    parser.add_argument(
        "--avg-span",
        action="store_true",
        help="average checkpoints across spans (span suffix removed from label)",
    )
    parser.add_argument(
        "--store-only",
        action="store_true",
        help="write JSON with --store and exit without plotting",
    )
    parser.add_argument(
        "--think-highlight",
        action="store_true",
        help="Use solid/dash-dot styles only for think traces; non-think traces become dotted",
    )
    parser.add_argument(
        "--think-xscale",
        type=float,
        default=1.0,
        help="Scale the X-axis of traces whose model name includes _thinkN by this factor",
    )
    parser.add_argument(
        "--think-yscale",
        type=float,
        default=1.0,
        help="Scale the Y-axis of traces whose model name includes _thinkN by this factor",
    )
    parser.add_argument(
        "--json-dir",
        type=pathlib.Path,
        default=pathlib.Path("sweep1"),
        help="Directory containing per-model JSON files from hist.py --write-json-dir",
    )
    return parser.parse_args()

def discover_paths(explicit: List[pathlib.Path], json_dir: pathlib.Path) -> List[pathlib.Path]:
    if explicit:
        collected: List[pathlib.Path] = []
        for item in explicit:
            if item.is_dir():
                collected.extend(sorted(item.glob("*.json")))
            else:
                collected.append(item)
        return collected
    return sorted(json_dir.glob("*.json"))


def main() -> None:
    args = parse_args()
    if args.default_test:
        if not args.train and not args.test:
            args.test = True
        args.no_nogrce = True
    if args.stored:
        records = load_store(args.stored)
    else:
        json_paths = discover_paths(args.paths, args.json_dir)
        records = load_records(json_paths)
        if not records and args.store and args.store.exists():
            records = load_store(args.store)
    if not records:
        print("No loss history found.")
        return

    if args.avg_span and records:
        grouped: dict[str, list[LossRecord]] = {}
        for rec in records:
            base_label = re.sub(r"_span\d+", "", rec.model_path.stem)
            grouped.setdefault(base_label, []).append(rec)

        averaged: List[LossRecord] = []
        for label, recs in grouped.items():
            base = recs[0]
            steps = base.steps

            def avg_series(attr: str) -> Optional[List[float]]:
                series = [getattr(r, attr) for r in recs]
                if any(s is None for s in series):
                    return None
                min_len = min(len(s) for s in series)
                return [sum(s[i] for s in series) / len(series) for i in range(min_len)]

            train_avg = avg_series("train")
            test_avg = avg_series("test")
            length = len(train_avg) if train_avg else len(test_avg) if test_avg else len(steps)

            averaged.append(
                LossRecord(
                    model_path=pathlib.Path(label),
                    steps=steps[:length],
                    train=train_avg or base.train,
                    test=test_avg or base.test,
                    train_nogrce=None,
                    test_nogrce=None,
                )
            )

        records = averaged

    if args.train_and_test:
        show_train = True
        show_test = True
    elif args.train:
        show_train = True
        show_test = False
    else:
        show_train = False
        show_test = True

    if args.store and records:
        store_records(
            records,
            args.store,
            include_train=show_train,
            include_test=show_test,
            include_nogrce=not args.no_nogrce,
        )
        if args.store_only:
            return

    steps_per_cycle = 100
    think_pattern = re.compile(r"_think\d+")

    def compute_scaled_steps(rec: LossRecord, *, is_think: bool) -> List[float]:
        if args.think_xscale != 1.0 and is_think:
            return [step * args.think_xscale for step in rec.steps]
        return list(rec.steps)

    def compute_scaled_series(
        series: Optional[List[float]], *, is_think: bool
    ) -> Optional[List[float]]:
        if series is None:
            return None
        if args.think_yscale != 1.0 and is_think:
            return [value * args.think_yscale for value in series]
        return list(series)

    for rec in records:
        is_think = bool(think_pattern.search(rec.model_path.stem))
        rec.is_think = is_think
        rec.scaled_steps = compute_scaled_steps(rec, is_think=is_think)
        rec.scaled_train = compute_scaled_series(rec.train, is_think=is_think)
        rec.scaled_test = compute_scaled_series(rec.test, is_think=is_think)
        rec.scaled_train_nogrce = compute_scaled_series(rec.train_nogrce, is_think=is_think)
        rec.scaled_test_nogrce = compute_scaled_series(rec.test_nogrce, is_think=is_think)

    steps_per_cycle = 100
    print(f"Loaded {len(records)} trace(s):")
    for rec in records:
        label = rec.model_path.stem
        scaled_steps = rec.scaled_steps
        if not scaled_steps:
            print(f"  - {label}: no sampled points")
            continue
        num_samples = len(scaled_steps)
        first_step = scaled_steps[0]
        last_step = scaled_steps[-1]
        cycles_start = first_step / steps_per_cycle
        cycles_end = last_step / steps_per_cycle
        parts: list[str] = []
        if rec.train:
            parts.append("train")
        if rec.test:
            parts.append("test")
        label_desc = ", ".join(parts) if parts else "no data"
        print(
            f"  - {label}: {num_samples} samples (steps {first_step}→{last_step}, cycles ~{cycles_start:.1f}→~{cycles_end:.1f}) [{label_desc}]"
        )
    fig, ax = plt.subplots(figsize=(12, 7))
    color_map: dict[str, str] = {}
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_index = 0
    for rec in records:
        label = rec.model_path.stem
        is_think = rec.is_think
        match = re.search(r"span(\d+)", label)
        span_key = match.group(1) if match else "ctx"
        if span_key not in color_map:
            color_map[span_key] = (
                "k" if span_key == "ctx" else default_colors[color_index % len(default_colors)]
            )
            color_index += span_key != "ctx"
        base_color = color_map[span_key]
        if args.think_highlight and not is_think:
            solid_style = ":"
        else:
            solid_style = "-"
        nogrce_style = "-."
        line_width = 3.0 if is_think else 1.5
        if show_train and rec.train:
            train_values = rec.scaled_train
            train_line, = ax.plot(
                rec.scaled_steps,
                train_values,
                label=f"{label} train",
                linestyle=solid_style,
                color=base_color,
                marker=None,
                linewidth=line_width,
            )
            if rec.train_nogrce and not args.no_nogrce:
                nogrce_values = rec.scaled_train_nogrce or rec.train_nogrce
                nogrce_line, = ax.plot(
                    rec.scaled_steps,
                    nogrce_values,
                    label=f"{label} train (nogrce)",
                    linestyle=nogrce_style,
                    color=base_color,
                    marker=None,
                    linewidth=line_width,
                )
        if show_test and rec.test:
            test_values = rec.scaled_test
            test_line, = ax.plot(
                rec.scaled_steps,
                test_values,
                label=f"{label} test",
                linestyle=solid_style,
                color=base_color,
                marker=None,
                linewidth=line_width,
            )
            if rec.test_nogrce and not args.no_nogrce:
                nogrce_test_values = rec.scaled_test_nogrce or rec.test_nogrce
                nogrce_line, = ax.plot(
                    rec.scaled_steps,
                    nogrce_test_values,
                    label=f"{label} test (nogrce)",
                    linestyle=nogrce_style,
                    color=base_color,
                    marker=None,
                    linewidth=line_width,
                )

    ax.set_xlabel("Steps")
    ylabel = "Loss" if show_train and show_test else ("Train Loss" if show_train else "Test Loss")
    ax.set_ylabel(ylabel)
    ax.set_title("Loss vs Steps")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
