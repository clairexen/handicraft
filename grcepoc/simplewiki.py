#!/usr/bin/env python3
"""Analyze line-level usage statistics for the SimpleWiki training corpus."""

from __future__ import annotations

import collections
import gzip
import pathlib
import re


def read_lines(path: pathlib.Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing corpus file: {path}")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return [line.rstrip("\n") for line in fh]
    return path.read_text(encoding="utf-8").splitlines()


def main() -> None:
    corpus_path = pathlib.Path("data") / "simplewiki-train.txt.gz"
    lines = read_lines(corpus_path)
    print(f"Loaded {len(lines):,} lines from {corpus_path}")
    word_re = re.compile(r"[a-z]+")
    word_counts: dict[str, int] = collections.Counter()
    for line in lines:
        words = set(word_re.findall(line.lower()))
        for word in words:
            word_counts[word] += 1
    if not word_counts:
        print("no words found")
        return
    sorted_words = sorted(word_counts.items(), key=lambda item: item[1])
    print("\nLeast popular words (line coverage):")
    for word, count in sorted_words[:10]:
        print(f"  {word:20s} {count:>8d}")
    print("\nMost popular words (line coverage):")
    for word, count in reversed(sorted_words[-10:]):
        print(f"  {word:20s} {count:>8d}")


if __name__ == "__main__":
    main()
