#!/usr/bin/env python3
"""Convert GSM8K-style JSONL.gz into the prompt/answer format expected by grce."""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable

SEPARATOR = "<|----|>"


def iter_records(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def format_entry(question: str, answer: str) -> str:
    return (
        "question:\n"
        f"{question.lower()}\n\n"
        "answer:\n"
        f"{answer.lower()}\n\n"
        f"{SEPARATOR}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a JSONL.GZ file containing question-answer records",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("-"),
        help="Optional output file (default: stdout)",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print dataset statistics (article count + token stats) instead of formatted entries",
    )
    args = parser.parse_args(argv)

    lengths: list[int] = []
    if args.count:
        for record in iter_records(args.input):
            question = record.get("question")
            answer = record.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                continue
            combined = f"{question}\n\n{answer}"
            lengths.append(len(combined.lower().split()))
        if not lengths:
            print("No valid records found.")
            return 0
        article_count = len(lengths)
        stats_min = min(lengths)
        stats_max = max(lengths)
        stats_mean = statistics.fmean(lengths)
        stats_median = statistics.median(lengths)
        print(f"articles: {article_count}")
        print(
            f"tokens - min: {stats_min}, max: {stats_max}, mean: {stats_mean:.2f}, median: {stats_median}"
        )
        thresholds = [64, 96, 128, 192, 256]
        for threshold in thresholds:
            below = sum(1 for length in lengths if length < threshold)
            pct = (below / article_count) * 100.0
            print(f"< {threshold} tokens: {pct:.2f}%")
        return 0

    if args.output == Path("-"):
        output_handle = sys.stdout
    else:
        output_handle = gzip.open(args.output, "wt", encoding="utf-8")
    try:
        for record in iter_records(args.input):
            question = record.get("question")
            answer = record.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                continue
            output_handle.write(format_entry(question, answer))
            output_handle.write("\n")
    finally:
        if output_handle is not sys.stdout:
            output_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
