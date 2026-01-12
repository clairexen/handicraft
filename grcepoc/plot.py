"""Plot train/test losses over steps for checkpoints."""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Iterable, List

import matplotlib.pyplot as plt


@dataclass
class LossRecord:
    model_path: pathlib.Path
    steps: List[int]
    train: List[float]
    test: List[float]


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
        records.append(LossRecord(pt_path, steps, train, test))
    return records


def torch_load(path: pathlib.Path):
    import torch

    return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=pathlib.Path,
        help=".pt checkpoint files (defaults to model/*.pt if omitted)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="plot train losses only")
    mode.add_argument("--test", action="store_true", help="plot test losses only")
    mode.add_argument("--both", action="store_true", help="plot both train and test losses")
    return parser.parse_args()


def discover_paths(explicit: List[pathlib.Path]) -> List[pathlib.Path]:
    if explicit:
        return explicit
    model_dir = pathlib.Path("model")
    return sorted(model_dir.glob("*.pt"))


def main() -> None:
    args = parse_args()
    pt_paths = discover_paths(args.paths)
    records = load_records(pt_paths)
    if not records:
        print("No loss history found.")
        return

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
    for rec in records:
        label = rec.model_path.stem
        if show_train:
            ax.plot(
                rec.steps,
                rec.train,
                label=f"{label} train",
                linestyle="--",
                marker=None
            )
        if show_test:
            ax.plot(
                rec.steps,
                rec.test,
                label=f"{label} test",
                linestyle="-",
                marker=None
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
