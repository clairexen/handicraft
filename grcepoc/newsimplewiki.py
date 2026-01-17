#!/usr/bin/env python3
"""Inspect paragraph lengths for the Simple English Wikipedia corpus."""
from __future__ import annotations

import argparse
import gzip
import pathlib
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt


@dataclass
class CorpusStats:
    name: str
    paragraphs: List[str]

    @property
    def lengths(self) -> List[int]:
        return [len(text) for text in self.paragraphs]

    def describe(self) -> str:
        lengths = self.lengths
        return (
            f"{self.name}: count={len(lengths)} min={min(lengths)} max={max(lengths)} "
            f"avg={sum(lengths) / len(lengths):.1f}"
        )


def load_corpus(path: pathlib.Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing corpus file: {path}")
    paragraphs: List[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            paragraphs.append(line.rstrip("\n"))
    return paragraphs


def build_vocab(paragraphs: List[str]) -> set[str]:
    vocab: set[str] = set()
    for text in paragraphs:
        for token in text.split():
            if token:
                vocab.add(token)
    return vocab


def unique_word_counts(paragraphs: List[str]) -> List[int]:
    word_occurrences: dict[str, int] = {}
    for text in paragraphs:
        tokens = {token for token in text.split() if token}
        for token in tokens:
            word_occurrences[token] = word_occurrences.get(token, 0) + 1
    counts: List[int] = []
    for text in paragraphs:
        tokens = {token for token in text.split() if token}
        counts.append(sum(1 for token in tokens if word_occurrences.get(token, 0) == 1))
    return counts


def plot_histogram(stats: List[CorpusStats], *, bins: int, output: pathlib.Path | None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for corpus in stats:
        lengths = corpus.lengths
        ax.hist(
            lengths,
            bins=bins,
            alpha=0.5,
            label=corpus.name,
        )
    ax.set_xlabel("Paragraph length (characters)")
    ax.set_ylabel("Count")
    ax.set_title("Simple English Wikipedia paragraph length distribution")
    ax.legend()
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=120)
        print(f"Saved histogram to {output}")
    else:
        plt.show()


def plot_unique_histogram(counts: List[int], *, bins: int, output: pathlib.Path | None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=bins, alpha=0.7, color="tab:orange")
    ax.set_xlabel("Unique words per paragraph")
    ax.set_ylabel("Count")
    ax.set_title("Paragraph-level unique word distribution (train)")
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=120)
        print(f"Saved unique-word histogram to {output}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument(
        "--plot-length-histogram",
        action="store_true",
        help="Generate the paragraph-length histogram plot",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional output PNG path (requires --plot)",
    )
    parser.add_argument(
        "--plot-unique-word-histogram",
        action="store_true",
        help="Generate histogram of unique word counts per paragraph",
    )
    parser.add_argument(
        "--unique-output",
        type=pathlib.Path,
        default=None,
        help="Optional output PNG for unique word plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.data / "simplewiki-train.txt.gz"
    test_path = args.data / "simplewiki-test.txt.gz"

    train_stats = CorpusStats("train", load_corpus(train_path))
    test_stats = CorpusStats("test", load_corpus(test_path))

    print(train_stats.describe())
    print(test_stats.describe())

    vocab = build_vocab(train_stats.paragraphs)
    print(f"train vocabulary size: {len(vocab)}")
    unique_counts = unique_word_counts(train_stats.paragraphs)
    unique_paragraphs = sum(1 for count in unique_counts if count > 0)
    print(
        "paragraphs containing a word unique to that paragraph: "
        f"{unique_paragraphs}/{len(train_stats.paragraphs)}"
    )

    if args.plot_length_histogram:
        plot_histogram([train_stats, test_stats], bins=args.bins, output=args.output)
    if args.plot_unique_word_histogram:
        plot_unique_histogram(unique_counts, bins=args.bins, output=args.unique_output)


if __name__ == "__main__":
    main()
