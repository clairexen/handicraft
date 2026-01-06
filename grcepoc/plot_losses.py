"""Plot train/test losses over steps for all checkpoints in model/."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt


@dataclass
class LossRecord:
    model_path: pathlib.Path
    steps: List[int]
    train: List[float]
    test: List[float]


def load_records(model_dir: pathlib.Path) -> List[LossRecord]:
    records = []
    for pt_path in sorted(model_dir.glob("*.pt")):
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


def main() -> None:
    model_dir = pathlib.Path("model")
    records = load_records(model_dir)
    if not records:
        print("No loss history found in model/.")
        return
    fig, ax = plt.subplots()
    for rec in records:
        label = rec.model_path.stem
        ax.plot(rec.steps, rec.test, label=f"{label} test", linestyle="-", marker="o")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Test Loss")
    ax.set_title("Test Loss vs Steps")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
