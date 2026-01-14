#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List


def slice_windows(series: List[float], steps_per_cycle: int, start: int, end: int) -> List[List[float]]:
    windows = []
    for w in range(start, end):
        begin = w * steps_per_cycle
        end_idx = begin + steps_per_cycle
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ctx vs. nogrce loss traces.")
    parser.add_argument("json", type=Path, help="JSON file produced via plot.py --store")
    parser.add_argument("--cycles", type=int, default=100, help="Steps per cycle (default: 100)")
    parser.add_argument("--spike-window", type=int, default=5, help="Steps after reset to measure spike amplitude")
    parser.add_argument("--start-window", type=int, default=0, help="First window index to analyze")
    parser.add_argument("--end-window", type=int, default=20, help="Window index to stop (exclusive)")
    parser.add_argument("--steady-start", type=int, default=10, help="Window index to treat as steady state")
    args = parser.parse_args()

    payload = json.loads(args.json.read_text())
    if isinstance(payload, list):
        payload = payload[0] if payload else {}

    steps = payload.get("steps", [])
    grce_train = payload.get("train")
    grce_test = payload.get("test")
    nogrce_train = payload.get("train_nogrce")
    nogrce_test = payload.get("test_nogrce")
    if not grce_train or not grce_test or nogrce_train is None or nogrce_test is None:
        raise SystemExit("JSON must contain train/test and train_nogrce/test_nogrce series")

    train_diff = [g - n for g, n in zip(grce_train, nogrce_train)]
    test_diff = [g - n for g, n in zip(grce_test, nogrce_test)]

    train_windows = slice_windows(train_diff, args.cycles, args.start_window, args.end_window)
    test_windows = slice_windows(test_diff, args.cycles, args.start_window, args.end_window)

    train_spikes = summarize_spikes(train_windows, args.spike_window)
    test_spikes = summarize_spikes(test_windows, args.spike_window)

    print("Train spike averages per window:", train_spikes)
    print("Test spike averages per window :", test_spikes)

    if len(train_spikes) >= 2:
        slope, intercept = np.polyfit(np.arange(len(train_spikes)), train_spikes, 1)
        print(f"Train spike slope per window: {slope:+.4f}")
    if len(test_spikes) >= 2:
        slope, intercept = np.polyfit(np.arange(len(test_spikes)), test_spikes, 1)
        print(f"Test spike slope per window : {slope:+.4f}")

    steady_idx = max(args.steady_start - args.start_window, 0)
    train_steady = train_windows[steady_idx:]
    test_steady = test_windows[steady_idx:]
    if train_steady:
        print("Steady-state train diff:", summarize_baseline(train_steady))
    if test_steady:
        print("Steady-state test diff :", summarize_baseline(test_steady))


if __name__ == "__main__":
    main()
