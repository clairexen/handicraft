#!/usr/bin/env python3
"""Analyze line-level usage statistics for the SimpleWiki training corpus."""

from __future__ import annotations

import argparse
import collections
import gzip
import math
import pathlib
import re


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
        return {}, 0, set()
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
        "--strip",
        type=int,
        default=1,
        help="Remove lines containing words that appear in <= N lines (default 1)",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=1,
        help="Stop when no word appears in <= N lines (default 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_path = pathlib.Path("data") / "simplewiki-train.txt.gz"
    lines = read_lines(corpus_path)
    if args.quick_test and len(lines) > 10_000:
        lines = lines[:10_000]
    print(f"Loaded {len(lines):,} lines from {corpus_path}")
    word_re = re.compile(r"[a-z]+")
    current_lines = list(lines)
    for iteration in range(1, 11):
        if not current_lines:
            print(f"\nIteration {iteration}: no lines remain, stopping early.")
            break
        print(f"\nIteration {iteration}: analyzing {len(current_lines):,} lines")
        word_counts = analyze_lines(current_lines, word_re=word_re)
        stop_words = {
            word for word, count in word_counts.items() if count <= max(1, args.stop)
        }
        if not stop_words:
            print(f"No words remain with <= {args.stop} lines; stopping.")
            break
        score_entries = []
        for idx, line in enumerate(current_lines):
            words = word_re.findall(line.lower())
            if not words:
                score = 0.0
            else:
                accum = 0.0
                for word in words:
                    count = word_counts.get(word, 1)
                    accum += math.log(max(count, 1)) ** 2
                score = math.sqrt(accum / len(words))
            score_entries.append((score, idx, line))
        score_entries.sort()
        strip_count = min(len(score_entries), max(1, args.strip))
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


if __name__ == "__main__":
    main()
