#!/usr/bin/env python3
"""Convert GSM8K-style JSONL.gz into the prompt/answer format expected by grce."""
from __future__ import annotations

import argparse
import gzip
import json
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
    args = parser.parse_args(argv)

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
