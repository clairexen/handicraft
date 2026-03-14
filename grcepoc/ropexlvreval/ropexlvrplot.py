from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import os
import re
from pathlib import Path
from typing import Iterable, List

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
FLOAT_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def load_column_losses(path: os.PathLike[str] | str) -> List[float]:
    """Return the list of per-column losses from a saved evaluation log."""

    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines: Iterable[str] = (ANSI_ESCAPE.sub("", line) for line in text.splitlines())
    collecting = False
    started = False
    values: List[float] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not collecting:
            if "Per-column losses:" in line:
                collecting = True
            continue
        if not started:
            if "[" not in line:
                continue
            started = True
        sanitized = line.replace("[", " ").replace("]", " ")
        for match in FLOAT_PATTERN.finditer(sanitized):
            values.append(float(match.group()))
        if "]" in line:
            break
    if not values:
        raise ValueError(f"could not locate per-column losses in {path}")
    return values

rope_losses = np.array(load_column_losses('eval_rope.txt'))
ropexl_losses = np.array(load_column_losses('eval_ropexl.txt'))
ropevr_losses = np.array(load_column_losses('eval_ropevr.txt'))
ropevrall_losses = np.array(load_column_losses('eval_ropevrall.txt'))

columns = np.arange(1, len(rope_losses) + 1)

smooth_factor = 16
if smooth_factor:
    rope_losses = rope_losses.reshape(-1, smooth_factor).mean(axis=1)
    ropexl_losses = ropexl_losses.reshape(-1, smooth_factor).mean(axis=1)
    ropevr_losses = ropevr_losses.reshape(-1, smooth_factor).mean(axis=1)
    ropevrall_losses = ropevrall_losses.reshape(-1, smooth_factor).mean(axis=1)
    columns = columns.reshape(-1, smooth_factor).mean(axis=1)

plt.figure(figsize=(12, 5))
if False:
    plt.title("Loss per column: RoPE-XL/RoPE-VR vs RoPE")
    plt.plot(columns, rope_losses, label="RoPE", color="#000", linestyle="--")
    plt.plot(columns, ropexl_losses, label="RoPE-XL", color="blue")
    plt.plot(columns, ropevr_losses, label="RoPE-VR", color="red")
    plt.plot(columns, ropevrall_losses, label="RoPE-VR-ALL", color="magenta")
else:
    plt.title("Loss per column: RoPE-XL/RoPE-VR relative to RoPE")
    plt.plot(columns, rope_losses - rope_losses, label="RoPE", color="#000", linestyle="--")
    plt.plot(columns, ropexl_losses - rope_losses, label="RoPE-XL", color="blue")
    plt.plot(columns, ropevr_losses - rope_losses, label="RoPE-VR", color="red")
    plt.plot(columns, ropevrall_losses - rope_losses, label="RoPE-VR-ALL", color="magenta")
plt.xlabel("Columns")
plt.ylabel("Loss Value (nats)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
