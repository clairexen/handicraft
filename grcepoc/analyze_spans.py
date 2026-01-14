#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def slice_windows(series: List[float], samples_per_cycle: int, start: int, end: int) -> List[List[float]]:
    windows = []
    for w in range(start, end):
        begin = w * samples_per_cycle
        end_idx = begin + samples_per_cycle
        windows.append(series[begin:end_idx])
    return windows


def summarize_spikes(windows: List[List[float]], spike_window: int) -> List[float]:
    spikes = []
    for win in windows:
        if len(win) < spike_window:
            continue
        spikes.append(sum(win[:spike_window]) / spike_window)
    return spikes


def summarize_baseline(windows: List[List[float]]) -> float:
    arr = [np.mean(win) for win in windows if win]
    return float(np.mean(arr)) if arr else float("nan")


def analyze_trace(record: dict, args: argparse.Namespace, label: str):
    steps = record.get("steps", [])
    grce_train = record.get("train")
    grce_test = record.get("test")
    grce_train_ng = record.get("train_nogrce")
    grce_test_ng = record.get("test_nogrce")
    if not grce_train or not grce_test:
        print(f"warning: record {label} missing train/test", file=sys.stderr)
        return [], [], float("nan"), float("nan")
    train_diff = (
        [g - n for g, n in zip(grce_train, grce_train_ng)]
        if grce_train_ng
        else grce_train
    )
    test_diff = (
        [g - n for g, n in zip(grce_test, grce_test_ng)]
        if grce_test_ng
        else grce_test
    )
    train_windows = slice_windows(
        train_diff, args.samples_per_cycle, args.start_window, args.end_window
    )
    test_windows = slice_windows(
        test_diff, args.samples_per_cycle, args.start_window, args.end_window
    )

    train_spikes = summarize_spikes(train_windows, args.spike_window)
    test_spikes = summarize_spikes(test_windows, args.spike_window)

    if train_spikes:
        print(f"{label} train spike averages per window: {train_spikes}")
    if test_spikes:
        print(f"{label} test spike averages per window : {test_spikes}")

    if len(train_spikes) >= 2:
        slope, intercept = np.polyfit(np.arange(len(train_spikes)), train_spikes, 1)
        print(f"{label} train spike slope per window: {slope:+.4f}")
    if len(test_spikes) >= 2:
        slope, intercept = np.polyfit(np.arange(len(test_spikes)), test_spikes, 1)
        print(f"{label} test spike slope per window : {slope:+.4f}")

    steady_idx = max(args.steady_start - args.start_window, 0)
    train_steady = train_windows[steady_idx:]
    test_steady = test_windows[steady_idx:]
    train_baseline = summarize_baseline(train_steady)
    test_baseline = summarize_baseline(test_steady)
    print(f"{label} steady-state train diff: {train_baseline}")
    print(f"{label} steady-state test diff : {test_baseline}")

    return train_spikes, test_spikes, train_baseline, test_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ctx vs. nogrce loss traces.")
    parser.add_argument("json", type=Path, help="JSON file produced via plot.py --store")
    parser.add_argument(
        "--samples-per-cycle",
        type=int,
        default=10,
        help="Number of plotted points per cycle (default: %(default)s)",
    )
    parser.add_argument(
        "--spike-window",
        type=int,
        default=5,
        help="Steps after reset to average for the spike metric (default: %(default)s)",
    )
    parser.add_argument(
        "--start-window",
        type=int,
        default=0,
        help="First window index to analyze (default: %(default)s)",
    )
    parser.add_argument(
        "--end-window",
        type=int,
        default=20,
        help="Exclusive upper window index (default: %(default)s)",
    )
    parser.add_argument(
        "--steady-start",
        type=int,
        default=10,
        help="Window index to treat as steady state (default: %(default)s)",
    )
    parser.add_argument("--plot", action="store_true", help="Display spike trends and baselines")
    args = parser.parse_args()

    payload = json.loads(args.json.read_text())
    if isinstance(payload, dict):
        records = [payload]
    else:
        records = payload

    fig = None
    ax = None
    if args.plot:
        fig, ax = plt.subplots()

    for idx, record in enumerate(records):
        label = record.get("model") or record.get("label") or f"record{idx}"
        train_spikes, test_spikes, train_base, test_base = analyze_trace(record, args, label)
        if args.plot and train_spikes:
            ax.plot(
                range(len(train_spikes)),
                train_spikes,
                label=f"{label} train",
                linestyle="-",
                marker="o",
            )
            if test_spikes:
                ax.plot(
                    range(len(test_spikes)),
                    test_spikes,
                    label=f"{label} test",
                    linestyle="--",
                    marker="x",
                )
    if args.plot and ax:
        ax.set_xlabel("Window index")
        ax.set_ylabel("Avg spike (loss diff)")
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
