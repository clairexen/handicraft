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


def unique_word_counts(paragraphs: List[str], indices: List[int]) -> dict[int, int]:
    word_occurrences: dict[str, int] = {}
    for idx in indices:
        tokens = {token for token in paragraphs[idx].split() if token}
        for token in tokens:
            word_occurrences[token] = word_occurrences.get(token, 0) + 1
    counts: dict[int, int] = {}
    for idx in indices:
        tokens = {token for token in paragraphs[idx].split() if token}
        counts[idx] = sum(1 for token in tokens if word_occurrences.get(token, 0) == 1)
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


def load_active_indices(path: pathlib.Path | None, total: int) -> List[int]:
    if path is None or not path.exists():
        return list(range(total))
    indices = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        value = int(line)
        if 0 <= value < total:
            indices.append(value)
    if not indices:
        return list(range(total))
    return sorted(set(indices))


def save_active_indices(path: pathlib.Path, indices: List[int]) -> None:
    path.write_text("\n".join(str(idx) for idx in sorted(indices)) + "\n", encoding="utf-8")
    print(f"Saved {len(indices)} active indices to {path}")


def drop_worst_paragraphs(
    paragraphs: List[str],
    indices: List[int],
    *,
    drop_count: int,
    iterations: int,
) -> List[int]:
    active = sorted(set(indices))
    drop_count = max(0, drop_count)
    iterations = max(0, iterations)
    if drop_count == 0 or iterations == 0:
        return active
    for step in range(iterations):
        if len(active) <= drop_count:
            break
        counts = unique_word_counts(paragraphs, active)
        ranked = sorted(active, key=lambda idx: (counts[idx], idx), reverse=True)
        to_remove = set(ranked[:drop_count])
        max_val = counts[ranked[0]] if ranked else 0
        active = [idx for idx in active if idx not in to_remove]
        remaining = len(active)
        updated_counts = unique_word_counts(paragraphs, active)
        unique_remaining = sum(1 for idx in active if updated_counts.get(idx, 0) > 0)
        percent = (unique_remaining / remaining * 100) if remaining else 0
        print(
            f"drop iteration {step + 1}: removed {len(to_remove)} paragraphs (max unique={max_val}); "
            f"unique paragraphs remaining: {unique_remaining}/{remaining} ({percent:.2f}%)"
        )
    return active


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
    parser.add_argument(
        "--drop-count",
        type=int,
        default=1000,
        help="Number of paragraphs to drop per iteration (by unique-word count)",
    )
    parser.add_argument(
        "--drop-iterations",
        type=int,
        default=10,
        help="How many drop iterations to perform (set 0 to disable)",
    )
    parser.add_argument(
        "--save-active",
        type=pathlib.Path,
        default=None,
        help="Write the remaining train paragraph indices after processing",
    )
    parser.add_argument(
        "--load-active",
        type=pathlib.Path,
        default=None,
        help="Read initial train paragraph indices (one per line)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.data / "simplewiki-train.txt.gz"
    test_path = args.data / "simplewiki-test.txt.gz"

    train_paragraphs = load_corpus(train_path)
    train_active = load_active_indices(args.load_active, len(train_paragraphs))
    if args.drop_count > 0 and args.drop_iterations > 0:
        train_active = drop_worst_paragraphs(
            train_paragraphs,
            train_active,
            drop_count=args.drop_count,
            iterations=args.drop_iterations,
        )
    train_subset = [train_paragraphs[idx] for idx in train_active]
    train_stats = CorpusStats("train", train_subset)
    test_stats = CorpusStats("test", load_corpus(test_path))

    print(train_stats.describe())
    print(test_stats.describe())

    vocab = build_vocab(train_stats.paragraphs)
    print(f"train vocabulary size: {len(vocab)}")
    unique_counts_map = unique_word_counts(train_paragraphs, train_active)
    unique_counts_list = [unique_counts_map[idx] for idx in train_active]
    unique_paragraphs = sum(1 for count in unique_counts_list if count > 0)
    total_active = len(train_active)
    percent_unique = (unique_paragraphs / total_active * 100) if total_active else 0
    print(
        "paragraphs containing a word unique to that paragraph: "
        f"{unique_paragraphs}/{total_active} ({percent_unique:.2f}%)"
    )

    if args.save_active:
        save_active_indices(args.save_active, train_active)

    if args.plot_length_histogram:
        plot_histogram([train_stats, test_stats], bins=args.bins, output=args.output)
    if args.plot_unique_word_histogram:
        plot_unique_histogram(
            unique_counts_list,
            bins=args.bins,
            output=args.unique_output,
        )


if __name__ == "__main__":
    main()
