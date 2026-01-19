#!/usr/bin/env python3
"""Analyze line-level usage statistics for the SimpleWiki training corpus."""

from __future__ import annotations

import argparse
import collections
import gzip
import math
import pathlib
import re

import matplotlib.pyplot as plt

def read_lines(path: pathlib.Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing corpus file: {path}")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return [line.rstrip("\n") for line in fh]
    return path.read_text(encoding="utf-8").splitlines()

def analyze_lines(
    lines: list[str],
    *,
    word_re: re.Pattern[str],
    top_k: int = 5,
) -> dict[str, int]:
    word_counts: dict[str, int] = collections.Counter()
    for line in lines:
        words = set(word_re.findall(line.lower()))
        for word in words:
            word_counts[word] += 1
    if not word_counts:
        return {}
    sorted_words = sorted(word_counts.items(), key=lambda item: item[1])
    print("Most popular words (line coverage):")
    for word, count in reversed(sorted_words[-top_k:]):
        print(f"  {word:20s} {count:>8d}")
    print("Least popular words (line coverage):")
    for word, count in sorted_words[:top_k]:
        print(f"  {word:20s} {count:>8d}")
    return word_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Only analyze the first 10,000 lines",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        help="Corpus base name inside data/ (uses <name>-train.txt.gz)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Operate on the test split instead of the train split",
    )
    parser.add_argument(
        "--strip",
        type=float,
        default=1.0,
        help="Percentage of lines to remove per iteration (default 1%%)",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=1,
        help="Stop when no word appears in <= N lines (default 1)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot histograms of word and line scores instead of shrinking",
    )
    parser.add_argument(
        "--loop",
        type=int,
        help="Number of shrink iterations to run (default: 0 = disabled)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_label = "test" if args.test else "train"
    if args.corpus:
        corpus_path = pathlib.Path("data") / f"{args.corpus}-{split_label}.txt.gz"
    else:
        corpus_path = pathlib.Path(f"shrink-{split_label}.txt")
    lines = read_lines(corpus_path)
    if args.quick_test and len(lines) > 10_000:
        lines = lines[:10_000]
    print(f"Loaded {len(lines):,} lines from {corpus_path}")
    word_re = re.compile(r"[a-z]+")
    scale = 100.0 / math.log(max(len(lines), 2))
    current_lines = list(lines)
    def compute_scores(lines: list[str], counts: dict[str, int]) -> tuple[list[float], list[float]]:
        word_scores = [math.log(max(count, 1)) * scale for count in counts.values()]
        line_scores: list[float] = []
        for line in lines:
            words = set(word_re.findall(line.lower()))
            if not words:
                continue
            accum = 0.0
            for word in words:
                count = counts.get(word, 1)
                weighted = math.log(max(count, 1)) * scale
                accum += weighted ** 2
            line_scores.append(math.sqrt(accum / len(words)))
        return word_scores, line_scores

    plot_before = None
    plot_after = None
    plot_path = pathlib.Path(f"shrink-{split_label}.png")
    if args.plot:
        initial_counts = analyze_lines(current_lines, word_re=word_re)
        if not initial_counts:
            print("No words found; nothing to plot.")
            return
        plot_before = compute_scores(current_lines, initial_counts)
        print(f"Initial vocabulary size: {len(initial_counts):,}")
    loop_iterations = 0
    if args.loop is not None:
        loop_iterations = max(0, args.loop)
    last_word_counts: dict[str, int] = {}
    if loop_iterations == 0:
        if args.plot and plot_before is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].hist(
                plot_before[0],
                bins=20,
                range=(0, 100),
                color="skyblue",
                edgecolor="black",
            )
            axes[0].set_title("Word score distribution (before)")
            axes[0].set_xlabel("100 * log(count) / log(total lines)")
            axes[0].set_ylabel("Frequency")
            axes[1].hist(
                plot_before[1],
                bins=20,
                range=(0, 100),
                color="salmon",
                edgecolor="black",
            )
            axes[1].set_title("Line score distribution (before)")
            axes[1].set_xlabel("RMS word score per line (scaled)")
            axes[1].set_ylabel("Frequency")
            fig.tight_layout()
            fig.savefig(plot_path)
            print(f"Saved plot to {plot_path}")
        output_path = pathlib.Path(f"shrink-{split_label}.txt")
        output_path.write_text("\n".join(current_lines) + "\n")
        print(f"\nWrote {len(current_lines):,} lines to {output_path}")
        return

    for iteration in range(1, loop_iterations + 1):
        if not current_lines:
            print(f"\nIteration {iteration}: no lines remain, stopping early.")
            break
        print(f"\nIteration {iteration}: analyzing {len(current_lines):,} lines")
        word_counts = analyze_lines(current_lines, word_re=word_re)
        print(f"Vocabulary size this iteration: {len(word_counts):,}")
        last_word_counts = word_counts
        stop_words = {
            word for word, count in word_counts.items() if count <= max(1, args.stop)
        }
        if not stop_words:
            print(f"No words remain with <= {args.stop} lines; stopping.")
            break
        score_entries = []
        for idx, line in enumerate(current_lines):
            words = set(word_re.findall(line.lower()))
            if not words:
                score = 0.0
            else:
                accum = 0.0
                for word in words:
                    count = word_counts.get(word, 1)
                    weighted = math.log(max(count, 1)) * scale
                    accum += weighted ** 2
                score = math.sqrt(accum / len(words))
            score_entries.append((score, idx, line))
        score_entries.sort()
        strip_fraction = max(0.0, args.strip)
        strip_count = max(1, int(len(score_entries) * strip_fraction / 100.0))
        strip_count = min(strip_count, len(score_entries))
        if strip_count <= 0:
            print("Strip count is zero; stopping.")
            break
        removed_indices = {score_entries[i][1] for i in range(strip_count)}
        removed_scores = [score_entries[i][0] for i in range(strip_count)]
        min_score = removed_scores[0]
        max_score = removed_scores[-1]
        print(
            f"Removing {strip_count} lowest-scoring lines: "
            f"scores {min_score:.4f} – {max_score:.4f}"
        )
        current_lines = [
            line
            for idx, line in enumerate(current_lines)
            if idx not in removed_indices
        ]

    output_path = pathlib.Path(f"shrink-{split_label}.txt")
    output_path.write_text("\n".join(current_lines) + "\n")
    print(f"\nWrote {len(current_lines):,} lines to {output_path}")
    if args.plot and plot_before is not None:
        plot_after = compute_scores(current_lines, last_word_counts)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0][0].hist(
            plot_before[0],
            bins=20,
            range=(0, 100),
            color="skyblue",
            edgecolor="black",
        )
        axes[0][0].set_title("Word score distribution (before)")
        axes[0][0].set_xlabel("100 * log(count) / log(total lines)")
        axes[0][0].set_ylabel("Frequency")
        axes[0][1].hist(
            plot_before[1],
            bins=20,
            range=(0, 100),
            color="salmon",
            edgecolor="black",
        )
        axes[0][1].set_title("Line score distribution (before)")
        axes[0][1].set_xlabel("RMS word score per line (scaled)")
        axes[0][1].set_ylabel("Frequency")
        axes[1][0].hist(
            plot_after[0],
            bins=20,
            range=(0, 100),
            color="skyblue",
            edgecolor="black",
        )
        axes[1][0].set_title("Word score distribution (after)")
        axes[1][0].set_xlabel("100 * log(count) / log(total lines)")
        axes[1][0].set_ylabel("Frequency")
        axes[1][1].hist(
            plot_after[1],
            bins=20,
            range=(0, 100),
            color="salmon",
            edgecolor="black",
        )
        axes[1][1].set_title("Line score distribution (after)")
        axes[1][1].set_xlabel("RMS word score per line (scaled)")
        axes[1][1].set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(plot_path)
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
