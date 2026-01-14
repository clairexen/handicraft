"""Plot train/test losses over steps for checkpoints."""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Optional

import json
import re

import matplotlib.pyplot as plt


@dataclass
class LossRecord:
    model_path: pathlib.Path
    steps: List[int]
    train: List[float]
    test: List[float]
    train_nogrce: Optional[List[float]] = None
    test_nogrce: Optional[List[float]] = None


def load_records(pt_paths: Iterable[pathlib.Path]) -> List[LossRecord]:
    records: List[LossRecord] = []
    for pt_path in pt_paths:
        if not pt_path.exists():
            print(f"warning: {pt_path} not found, skipping")
            continue
        data = torch_load(pt_path)
        history = data.get("loss_history", []) if isinstance(data, dict) else []
        if not history:
            continue
        steps = [int(item.get("step", i + 1)) for i, item in enumerate(history)]
        train = [float(item.get("train_loss", float("nan"))) for item in history]
        test = [float(item.get("test_loss", float("nan"))) for item in history]
        train_ng: List[float] = []
        test_ng: List[float] = []
        has_train_ng = True
        has_test_ng = True
        for item in history:
            val = item.get("train_loss_nogrce")
            if val is None:
                has_train_ng = False
                break
            train_ng.append(float(val))
        for item in history:
            val = item.get("test_loss_nogrce")
            if val is None:
                has_test_ng = False
                break
            test_ng.append(float(val))
        records.append(
            LossRecord(
                pt_path,
                steps,
                train,
                test,
                train_nogrce=train_ng if has_train_ng else None,
                test_nogrce=test_ng if has_test_ng else None,
            )
        )
    return records


def torch_load(path: pathlib.Path):
    import torch

    return torch.load(path, map_location="cpu")


def load_store(path: pathlib.Path) -> Optional[LossRecord]:
    if not path.exists():
        print(f"warning: stored file {path} not found")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    return LossRecord(
        model_path=path,
        steps=list(map(int, payload.get("steps", []))),
        train=list(map(float, payload.get("train", []))),
        test=list(map(float, payload.get("test", []))),
        train_nogrce=(
            list(map(float, payload.get("train_nogrce", [])))
            if payload.get("train_nogrce")
            else None
        ),
        test_nogrce=(
            list(map(float, payload.get("test_nogrce", [])))
            if payload.get("test_nogrce")
            else None
        ),
    )


def store_records(records: List[LossRecord], out_path: pathlib.Path) -> None:
    payload = [
        {
            "model": rec.model_path.name,
            "steps": rec.steps,
            "train": rec.train,
            "test": rec.test,
            **(
                {"train_nogrce": rec.train_nogrce}
                if rec.train_nogrce is not None
                else {}
            ),
            **(
                {"test_nogrce": rec.test_nogrce}
                if rec.test_nogrce is not None
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
    parser.add_argument("--test", action="store_true", help="plot test losses only")
    parser.add_argument("--both", action="store_true", help="plot both train and test losses")
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
        "--avg-span",
        action="store_true",
        help="average checkpoints across spans (span suffix removed from label)",
    )
    return parser.parse_args()

def discover_paths(explicit: List[pathlib.Path]) -> List[pathlib.Path]:
    if explicit:
        return explicit
    model_dir = pathlib.Path("model")
    return sorted(model_dir.glob("*.pt"))


def main() -> None:
    args = parse_args()
    if args.stored:
        record = load_store(args.stored)
        records = [record] if record else []
    else:
        pt_paths = discover_paths(args.paths)
        records = load_records(pt_paths)
        if not records and args.store and args.store.exists():
            record = load_store(args.store)
            records = [record] if record else []
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

            averaged.append(
                LossRecord(
                    model_path=pathlib.Path(label),
                    steps=steps[: len(train_avg) if train_avg else len(steps)],
                    train=train_avg or base.train,
                    test=test_avg or base.test,
                    train_nogrce=avg_series("train_nogrce"),
                    test_nogrce=avg_series("test_nogrce"),
                )
            )

        records = averaged

    if args.store and records:
        store_records(records, args.store)

    if args.both:
        show_train = show_test = True
    elif args.train:
        show_train, show_test = True, False
    elif args.test:
        show_train, show_test = False, True
    else:
        show_test = True
        show_train = len(records) == 1

    fig, ax = plt.subplots()
    color_map: dict[str, str] = {}
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_index = 0
    for rec in records:
        label = rec.model_path.stem
        match = re.search(r"span(\d+)", label)
        span_key = match.group(1) if match else "ctx"
        if span_key not in color_map:
            color_map[span_key] = (
                "k" if span_key == "ctx" else default_colors[color_index % len(default_colors)]
            )
            color_index += span_key != "ctx"
        base_color = color_map[span_key]
        if show_train:
            ax.plot(
                rec.steps,
                rec.train,
                label=f"{label} train",
                linestyle="-",
                color=base_color,
                marker=None,
            )
            if rec.train_nogrce and not args.no_nogrce:
                ax.plot(
                    rec.steps,
                    rec.train_nogrce,
                    label=f"{label} train (nogrce)",
                    linestyle="-.",
                    color=base_color,
                    marker=None,
                )
        if show_test:
            ax.plot(
                rec.steps,
                rec.test,
                label=f"{label} test",
                linestyle="-",
                color=base_color,
                marker=None,
            )
            if rec.test_nogrce and not args.no_nogrce:
                ax.plot(
                    rec.steps,
                    rec.test_nogrce,
                    label=f"{label} test (nogrce)",
                    linestyle="-.",
                    color=base_color,
                    marker=None,
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
