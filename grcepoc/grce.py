#
# GPT with GRCE (Gradient-limited Recurrent Context Encoding) and XCTX
#
# Copyright (C) 2026  Claire Xenia Wolf <claire@clairexen.net>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""GRCE proof-of-concept.

This module combines a picoGPT-style Transformer with Gradient-limited
Recurrent Context Encoding (GRCE) and XCTX channels. It exposes the full CLI,
training loop, evaluation utilities, and reporting helpers used by the
``grce.py`` entry point. Key entities:

* :class:`Args` – runtime configuration passed into tokenizer/model builders,
  :func:`describe_model_size`, and :func:`grce_main`.
* :func:`grce_cli_args` – constructs the CLI parser; invoked at startup and by
external tooling to mirror the binary interface. Its result is consumed by
:func:`grce_main`.
* :func:`train_model` – the main training loop used by :func:`grce_main`. It
handles batching, diagnostics, and logging.
* :func:`evaluate_single_split_batch` / :func:`evaluate_single_stacked_batch`
  – evaluate batch losses for the split/stacked layouts.
* :func:`describe_model_size` – backs the ``size`` subcommand using
  :class:`ModelGeometry` metadata.

Call tree (simplified)::

    grce_cli_args
        └── grce_main
            ├── train_model
            │     └── evaluate_single_*_batch
            └── describe_model_size
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# File layout overview
# -----------------------------------------------------------------------------
# This module is intentionally split into seven large blocks so lightweight
# tooling can inspect CLI options without importing PyTorch:
#   1. GRCE Model Configuration  – dataclass definitions and static defaults.
#   2. GRCE CLI Argument Parser  – argparse setup plus the first
#      `if __name__ == "__main__"` which only parses CLI flags and exits for
#      requests such as `--help` before heavy imports occur.
#   3. GRCE Library Components   – tokenizer helpers, shared utilities, and
#      support code that depends on PyTorch/tokenizers.
#   4. Data Utilities            – dataset wrappers, prompt trackers, and
#      corpus helpers.
#   5. GRCE Model Components     – Transformer/GRCE modules built on torch.nn.
#   6. Training / Generation Helpers – augmentation logic, evaluation helpers,
#      and the optimizer/training loop.
#   7. GRCE CLI Main Function    – end-to-end orchestration plus the second
#      `if __name__ == "__main__"` which imports torch (while running block 3),
#      executes `grce_main`, and returns an exit status.
# The dual entry points are deliberate: parsing the CLI is “cheap” and resides
# entirely above the heavy imports, while the final entry point performs the
# full setup once we know we actually need to train/evaluate.


PROMPT_GOALS = [
    ("one plus one is", " two"),
    ("fire is hot and ice is", " cold"),
    ("the opposite of up is", " down"),
    ("ice is cold and fire is", " hot"),
    ("the first letter of the alphabet is", " a"),
    ("the first letter of the alphabet is the letter", " a"),
    ("the color of a red apple is", " red"),
    ("the sun rises in the", " east"),
    ("the sun sets in the", " west"),
    ("earth's satellite is the", " moon"),
    ("a baby dog is called a puppy and a baby cat is called a", " kitten"),
    ("a baby cat is called a kitten and a baby dog is called a", " puppy"),
    ("celsius is based on the freezing and boiling points of water. water freezes at", " 0"),
]


# -----------------------------------------------------------------------------
# GRCE Model Configuration
# -----------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class ModelGeometry:
    """Holds the GPT+GRCE+XCTX model geometry."""

    vocab_size: int = 6000  # GPT-2 base supports ~50k merges; we stay small for the PoC.
    block_size: int = 256   # GPT-2 base uses 1024 tokens.
    n_layer: int = 8        # GPT-2 base uses 12 layers.
    n_head: int = 6         # GPT-2 base uses 12 attention heads.
    n_embd: int = 384       # GPT-2 base uses 768 embedding dims.
    n_grce: int = 64        # Narrow GRCE context dims.
    n_xctx: int = 720       # Wide XCTX context dims.

MODEL_GEOMETRY_DEFAULTS = ModelGeometry()


@dataclass
class Defaults:
    """Default Settings (override with CLI args)"""

    vocab_size: int = MODEL_GEOMETRY_DEFAULTS.vocab_size
    block_size: int = MODEL_GEOMETRY_DEFAULTS.block_size
    n_layer: int = MODEL_GEOMETRY_DEFAULTS.n_layer
    n_head: int = MODEL_GEOMETRY_DEFAULTS.n_head
    n_embd: int = MODEL_GEOMETRY_DEFAULTS.n_embd
    n_grce: int = MODEL_GEOMETRY_DEFAULTS.n_grce
    n_xctx: int = MODEL_GEOMETRY_DEFAULTS.n_xctx
    corpus: str | None = None
    steps: int = 100
    cycles: int = 100
    batch_size: int = 256
    layout: str = "2[*d]+2[*f],*[*1-2e/*1-4d/*1-4f/*1-2n],*[*1-2e/*1-4f/*1-4d/*1-2n]"
    eval_interval: int = 10
    dropout: float = 0.05
    detach_span: int = 0
    log_step_details: bool = False

DEFAULTS = Defaults()


from argparse import Namespace as Args
GeometryLike = ModelGeometry | Args


FANCY_SPACE = "\u2423"  # Open Box symbol for visible spaces
FANCY_ENTER = "\u23CE " # Return symbol for visible newlines

def normalize_prompt(text: str) -> str:
    """Map placeholder characters back to literal spaces/newlines."""

    return text.replace(FANCY_SPACE, " ").replace(FANCY_ENTER, "\n"). \
            replace(FANCY_ENTER.replace(" ", "\n"), "\n")


# -----------------------------------------------------------------------------
# GRCE Layout String Parser and Interpreter
# -----------------------------------------------------------------------------
# 
# Mini-language parser for GRCE batch layouts.
# 
# This module hosts :class:`BatchLayout`, a helper that interprets the batch
# layout strings described in the project notes.  The class handles the
# preprocessor (``(...)`` alternatives), row/column range specs, ``*`` expansion
# markers, and exposes the resolved concrete layout plus serialization helpers.
# 
# The implementation mirrors the specification in the user instructions:
# 
# * ``A+B+...`` concatenates row groups; each group is ``ROWS[SEGMENTS]``.
# * ``SEGMENTS`` are slash-separated ``COLS``+``MODE`` tokens (``16e/32d``).
# * Ranges use ``START-END`` (inclusive) and may be prefixed by ``*`` or ``+`` to allow
#   dynamic expansion when additional rows/columns are needed.
# * Bare ``*`` behaves like ``*0-0`` for the baseline and participates in expansion with
#   weight ``max(1, current_size)``.
# * ``+N-M`` behaves like ``*N-M`` but never grows beyond ``M`` (``+M`` means ``+1-M`` and
#   bare ``+`` expands with a constant weight of 1).
# * Parentheses with ``|`` act as a textual pre-processor: ``A(B|C)D`` randomly
#   expands to either ``ABD`` or ``ACD`` before the parser runs.
# 
# The resolved layout is stored as structured data and can be re-serialized into
# the concrete (fully deterministic) layout string.

from dataclasses import dataclass
import random
from typing import List, Sequence


_MODE_ALIASES: dict[str, str] = {
    "e": "encode",
    "d": "decode",
    "f": "forward",
    "n": "noattn",
}

_MODE_LETTERS: dict[str, str] = {value: key for key, value in _MODE_ALIASES.items()}


class LayoutParseError(ValueError):
    """Raised when a layout string cannot be parsed."""


def _preprocess_template(template: str, rng: random.Random) -> str:
    """Expand ``(...)`` alternatives by randomly choosing one branch."""

    def parse_group(index: int) -> tuple[str, int]:
        options: List[str] = []
        current: List[str] = []
        while index < len(template):
            ch = template[index]
            if ch == "(":
                chunk, index = parse_group(index + 1)
                current.append(chunk)
                continue
            if ch == ")":
                index += 1
                options.append("".join(current))
                if not options:
                    raise LayoutParseError("Empty () block in layout template")
                choice = rng.choice(options)
                return choice, index
            if ch == "|":
                options.append("".join(current))
                current.clear()
                index += 1
                continue
            current.append(ch)
            index += 1
        raise LayoutParseError("Unbalanced '(' in layout template")

    output: List[str] = []
    index = 0
    while index < len(template):
        ch = template[index]
        if ch == "(":
            chunk, index = parse_group(index + 1)
            output.append(chunk)
            continue
        if ch == ")":
            raise LayoutParseError("Unbalanced ')' in layout template")
        output.append(ch)
        index += 1
    return "".join(output)


@dataclass
class CountSpec:
    """Specification for row/column counts with optional expansion markers."""

    minimum: int
    maximum: int
    expandable: bool
    constant_weight: bool = False
    cap: int | None = None

    @classmethod
    def parse(cls, token: str) -> "CountSpec":
        token = token.strip()
        if not token:
            raise LayoutParseError("Missing size spec")
        prefix = None
        if token.startswith("*"):
            prefix = "*"
            token = token[1:]
        elif token.startswith("+"):
            prefix = "+"
            token = token[1:]
        expandable = prefix in {"*", "+"}
        constant_weight = prefix == "+"
        cap: int | None = None
        if prefix == "+" and token and "-" not in token:
            minimum = 1
            maximum = int(token)
            cap = maximum
        elif not token:
            minimum = maximum = 0
        else:
            if "-" in token:
                parts = token.split("-", 1)
                if len(parts) != 2:
                    raise LayoutParseError(f"Invalid range '{token}'")
                minimum = int(parts[0])
                maximum = int(parts[1])
            else:
                minimum = maximum = int(token)
            if prefix == "+":
                cap = maximum
        if minimum < 0 or maximum < 0:
            raise LayoutParseError("Negative sizes are not supported")
        if maximum < minimum:
            raise LayoutParseError(f"Invalid range {minimum}-{maximum}")
        return cls(minimum, maximum, expandable, constant_weight, cap)

    def sample(self, rng: random.Random) -> int:
        if self.minimum == self.maximum:
            return self.minimum
        return rng.randint(self.minimum, self.maximum)


@dataclass
class SegmentSpec:
    size: CountSpec
    mode: str


@dataclass
class RowSpec:
    count: CountSpec
    segments: list[SegmentSpec]


@dataclass
class SegmentLayout:
    mode: str
    columns: int


@dataclass
class RowLayout:
    rows: int
    segments: list[SegmentLayout]

    def total_columns(self) -> int:
        return sum(segment.columns for segment in self.segments)

    def token_span(self) -> int:
        columns = self.total_columns()
        if columns <= 0 or self.rows <= 0:
            return 0
        return self.rows * (columns + 1)


def _split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    depth_square = 0
    depth_paren = 0
    start = 0
    for index, ch in enumerate(text):
        if ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1
            if depth_square < 0:
                raise LayoutParseError("Unbalanced ']' in layout string")
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
            if depth_paren < 0:
                raise LayoutParseError("Unbalanced ')' in layout string")
        elif ch == sep and depth_square == 0 and depth_paren == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    if depth_square != 0:
        raise LayoutParseError("Unbalanced '[' in layout string")
    if depth_paren != 0:
        raise LayoutParseError("Unbalanced '(' in layout string")
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _parse_row_spec(token: str) -> RowSpec:
    token = token.strip()
    if not token:
        raise LayoutParseError("Missing row definition")
    bracket = token.find("[")
    if bracket <= 0 or not token.endswith("]"):
        raise LayoutParseError(f"Invalid row term '{token}'")
    count_token = token[: bracket].strip()
    segments_body = token[bracket + 1 : -1]
    if not count_token:
        raise LayoutParseError("Row group missing row count")
    count_spec = CountSpec.parse(count_token)
    if not segments_body:
        raise LayoutParseError("Row group requires at least one segment")
    segment_tokens = _split_segments(segments_body)
    segments = [_parse_segment_spec(item) for item in segment_tokens]
    return RowSpec(count_spec, segments)


def _split_segments(body: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for index, ch in enumerate(body):
        if ch in "[]()":
            raise LayoutParseError("Unexpected bracket in segment string")
        if ch == "/":
            parts.append(body[start:index].strip())
            start = index + 1
    parts.append(body[start:].strip())
    return [part for part in parts if part]


def _parse_segment_spec(text: str) -> SegmentSpec:
    text = text.strip()
    if not text:
        raise LayoutParseError("Empty segment definition")
    index = 0
    while index < len(text) and not text[index].isalpha():
        index += 1
    if index == len(text):
        raise LayoutParseError(f"Missing mode in segment '{text}'")
    size_token = text[:index]
    mode_token = text[index:].lower()
    if mode_token not in _MODE_ALIASES:
        raise LayoutParseError(f"Unsupported mode '{mode_token}'")
    size_spec = CountSpec.parse(size_token or "1")
    return SegmentSpec(size_spec, _MODE_ALIASES[mode_token])


@dataclass
class _CountAllocation:
    spec: CountSpec
    value: int

    def can_shrink(self) -> bool:
        return self.value > self.spec.minimum

    def shrink(self) -> None:
        if not self.can_shrink():
            raise ValueError("Cannot shrink below minimum")
        self.value -= 1

    def can_expand(self) -> bool:
        if not self.spec.expandable:
            return False
        if self.spec.cap is not None and self.value >= self.spec.cap:
            return False
        return True

    def expand_weight(self) -> int:
        if not self.can_expand():
            return 0
        if self.spec.constant_weight:
            return 1
        return max(1, self.value)

    def expand(self) -> None:
        if not self.can_expand():
            raise ValueError("Cannot expand fixed allocation")
        self.value += 1


def _shrink_until(target: int, items: Sequence[_CountAllocation], rng: random.Random) -> bool:
    current = sum(item.value for item in items)
    while current > target:
        candidates = [item for item in items if item.can_shrink()]
        if not candidates:
            return False
        choice = rng.choice(candidates)
        choice.shrink()
        current -= 1
    return True


def _expand_until(target: int, items: Sequence[_CountAllocation], rng: random.Random) -> bool:
    current = sum(item.value for item in items)
    while current < target:
        candidates = [item for item in items if item.can_expand()]
        if not candidates:
            return False
        weights = [item.expand_weight() for item in candidates]
        choice = rng.choices(candidates, weights=weights, k=1)[0]
        choice.expand()
        current += 1
    return True


class BatchLayout:
    """Concrete representation of a parsed batch layout."""

    def __init__(
        self,
        template: str,
        batch_size: int,
        block_size: int,
        *,
        rng: random.Random | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.template = template
        self.batch_size = batch_size
        self.block_size = block_size
        self.rng = rng or random
        self.warnings: list[str] = []
        processed = _preprocess_template(template, self.rng)
        micro_terms = _split_top_level(processed.replace(" ", ""), ",")
        if not micro_terms:
            raise LayoutParseError("Layout string is empty")
        self.micro_specs: list[list[RowSpec]] = []
        for micro_term in micro_terms:
            if not micro_term:
                continue
            row_terms = _split_top_level(micro_term, "+")
            if not row_terms:
                continue
            self.micro_specs.append([_parse_row_spec(term) for term in row_terms])
        if not self.micro_specs:
            raise LayoutParseError("Layout string is empty")
        self.micro_batches = [self._materialize_rows(specs) for specs in self.micro_specs]
        self.rows = [row for batch in self.micro_batches for row in batch]

    def micro_token_spans(self) -> list[int]:
        spans: list[int] = []
        for batch in self.micro_batches:
            span = sum(row.token_span() for row in batch)
            spans.append(span)
        return spans

    def total_token_span(self) -> int:
        return sum(self.micro_token_spans())

    def _materialize_rows(self, specs: Sequence[RowSpec]) -> list[RowLayout]:
        row_allocs: list[_CountAllocation] = []
        for spec in specs:
            value = spec.count.sample(self.rng)
            row_allocs.append(_CountAllocation(spec.count, value))
        if not _shrink_until(self.batch_size, row_allocs, self.rng):
            min_rows = sum(item.spec.minimum for item in row_allocs)
            self.warnings.append(
                f"Row count lower bounds ({min_rows}) exceed batch size ({self.batch_size})"
            )
        else:
            _expand_until(self.batch_size, row_allocs, self.rng)
        rows: list[RowLayout] = []
        for spec, allocation in zip(specs, row_allocs):
            segments = self._materialize_segments(spec.segments)
            rows.append(RowLayout(allocation.value, segments))
        total_rows = sum(row.rows for row in rows)
        if total_rows > self.batch_size:
            self.warnings.append(
                f"Resolved layout uses {total_rows} rows which exceeds batch size {self.batch_size}"
            )
        return rows

    def _materialize_segments(self, specs: Sequence[SegmentSpec]) -> list[SegmentLayout]:
        allocations: list[_CountAllocation] = []
        for spec in specs:
            value = spec.size.sample(self.rng)
            allocations.append(_CountAllocation(spec.size, value))
        if not _shrink_until(self.block_size, allocations, self.rng):
            min_cols = sum(item.spec.minimum for item in allocations)
            self.warnings.append(
                f"Segment lower bounds ({min_cols}) exceed block size ({self.block_size})"
            )
        else:
            _expand_until(self.block_size, allocations, self.rng)
        segments = [SegmentLayout(spec.mode, allocation.value) for spec, allocation in zip(specs, allocations)]
        max_cols = sum(segment.columns for segment in segments)
        if max_cols > self.block_size:
            self.warnings.append(
                f"Row with modes {[segment.mode for segment in segments]} exceeds block size"
            )
        return segments

    def serialize(self) -> str:
        """Return a deterministic layout string for the resolved layout."""

        micro_parts = [self.serialize_rows(batch) for batch in self.micro_batches]
        return ",".join(micro_parts)

    def serialize_rows(self, rows: Sequence[RowLayout]) -> str:
        row_bits: list[str] = []
        for row in rows:
            segment_bits = []
            for segment in row.segments:
                letter = _MODE_LETTERS.get(segment.mode, segment.mode[0])
                segment_bits.append(f"{segment.columns}{letter}")
            row_bits.append(f"{row.rows}[{'/'.join(segment_bits)}]")
        return "+".join(row_bits)

    def expanded_rows(self) -> list[list[SegmentLayout]]:
        """Return the per-row segments with rows fully expanded."""

        rows: list[list[SegmentLayout]] = []
        for group in self.rows:
            for _ in range(group.rows):
                rows.append([SegmentLayout(seg.mode, seg.columns) for seg in group.segments])
        return rows

# -----------------------------------------------------------------------------
# GRCE CLI Argument Parser
# -----------------------------------------------------------------------------

import argparse
import math
import os
import pathlib
import random
import re
import shlex
import sys
import time
from collections import OrderedDict, defaultdict
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Sequence, Callable


def grce_cli_args(argv: Sequence[str] | None = None) -> Args:
    """Parse CLI arguments and return the populated namespace.

    Used by the ``if __name__ == '__main__'`` entry point and by tooling that
    wants to mirror the CLI behavior without invoking the binary. The result
    is passed directly to :func:`grce_main`.
    """
    if argv is None:
        raw_cli_args = sys.argv[1:]
    elif argv is sys.argv:
        raw_cli_args = list(argv[1:])
    else:
        raw_cli_args = list(argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    def flag_present(flag: str) -> bool:
        return any(arg == flag or arg.startswith(f"{flag}=") for arg in raw_cli_args)
    generic = parser.add_argument_group("Generic options")
    generic.add_argument(
        "--name",
        type=str,
        default="default",
        help="Run identifier used when naming checkpoints; defaults to 'default'.",
    )
    generic.add_argument(
        "--data",
        type=str,
        default="data",
        help="Directory containing <corpus>-train.txt.gz and <corpus>-test.txt.gz",
    )
    generic.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer JSON path overriding checkpoints and cached files",
    )
    generic.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    generic.add_argument("--torch-compile", type=str, default="off", help="off or default or reduce-overhead")
    generic.add_argument(
        "--model",
        type=str,
        default="model",
        help="Directory where checkpoints/logs/tokenizers are stored",
    )

    model_group = parser.add_argument_group("Model configuration")
    model_group.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULTS.vocab_size,
        help="Total vocabulary size for the tokenizer (including special tokens)",
    )
    model_group.add_argument(
        "--block-size",
        type=int,
        default=DEFAULTS.block_size,
        help="Maximum sequence length supported by the model's positional embeddings",
    )
    model_group.add_argument(
        "--block-length",
        type=int,
        default=None,
        help="Actual tokens-per-sample used during train/eval (defaults to --block-size)",
    )
    model_group.add_argument(
        "--n-layer",
        type=int,
        default=DEFAULTS.n_layer,
        help="Number of transformer blocks (GPT-2 base uses 12).",
    )
    model_group.add_argument(
        "--n-head",
        type=int,
        default=DEFAULTS.n_head,
        help="Number of attention heads per block (GPT-2 base uses 12).",
    )
    model_group.add_argument(
        "--n-embd",
        type=int,
        default=DEFAULTS.n_embd,
        help="Embedding/hidden dimension (GPT-2 base uses 768); must be a multiple of n_head.",
    )
    model_group.add_argument(
        "--n-grce",
        type=int,
        default=DEFAULTS.n_grce,
        help="Dimension of the recurrent GRCE context; use 0 to disable the channel.",
    )
    model_group.add_argument(
        "--grce-optimized",
        action="store_true",
        help="Use the vectorized GRCE channel implementation (experimental).",
    )
    model_group.add_argument(
        "--n-xctx",
        type=int,
        default=DEFAULTS.n_xctx,
        help=(
            "Dimension of the wide (layer-partitioned) context channel; must be a multiple of n_layer"
        ),
    )
    model_group.add_argument(
        "--tiny",
        action="store_true",
        help=(
            "Shortcut for --vocab-size 600 --batch-size 12 --block-size 6 --n-layer 3 --n-head 2 "
            "--n-embd 8 --n-grce 4 --n-xctx 9 --steps 2 --cycles 1 --eval-interval 1"
        ),
    )

    training_group = parser.add_argument_group("Training schedule")
    training_group.add_argument("--steps", type=int, default=DEFAULTS.steps, help="Training steps per cycle")
    training_group.add_argument(
        "--cycles",
        type=int,
        default=DEFAULTS.cycles,
        help="Repeat the full training/eval/update cycle N times.",
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULTS.batch_size,
        help="Number of sequences per optimization step.",
    )
    training_group.add_argument(
        "--layout",
        type=str,
        default=DEFAULTS.layout,
        help=(
            "Batch layout mini-language string controlling per-row encode/decode/forward/noattn segments."
            " Supports ranges, '*' expansions, and () alternations."
        ),
    )
    training_group.add_argument(
        "--log-step-details",
        action="store_true",
        default=DEFAULTS.log_step_details,
        help="Print per-step micro-batch timing details",
    )
    training_group.add_argument(
        "--no-grad-summary",
        action="store_true",
        help="Disable per-cycle gradient summary logging",
    )
    training_group.add_argument(
        "--log-grad-norms",
        action="store_true",
        help="Print per-micro and per-step gradient norms each step",
    )
    training_group.add_argument(
        "--skip-model-update",
        action="store_true",
        help="Skip overwriting the checkpoint at the end of each training cycle",
    )
    training_group.add_argument(
        "--restart-optimizer",
        action="store_true",
        help="Reinitialize the optimizer at the beginning of every cycle",
    )
    training_group.add_argument(
        "--checkpoint-optimizer",
        action="store_true",
        help="Serialize optimizer state to checkpoints so runs can resume without momentum reset",
    )
    training_group.add_argument(
        "--no-kv-rebalance",
        action="store_true",
        help="Disable binary-segment merging in KV caches (debug/perf testing)",
    )
    training_group.add_argument(
        "--detach-kv-cache",
        action="store_true",
        help=(
            "Detach KV cache tensors into a preallocated buffer so incremental decoding doesn't"
            " backprop through previous steps"
        ),
    )
    training_group.add_argument(
        "--detach-span",
        type=int,
        default=DEFAULTS.detach_span,
        help="Detach GRCE context gradients every N positions (0 disables detaching).",
    )
    training_group.add_argument(
        "--no-detach-ctx",
        action="store_true",
        help="Keep gradients through the recurrent GRCE context even when spans trigger",
    )
    training_group.add_argument(
        "--dropout",
        type=float,
        default=DEFAULTS.dropout,
        help="Dropout probability inside attention/FFN blocks.",
    )
    training_group.add_argument(
        "--detach-layer",
        type=int,
        default=-1,
        help="If >0, detach gradients after this Transformer layer (1-based index).",
    )
    training_group.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="How often to run train/test evaluation steps.",
    )


    sampling_group = parser.add_argument_group("Sampling & reporting")
    sampling_group.add_argument(
        "--prompt",
        type=str,
        default="ai will",
        help="Prompt used for generation",
    )
    sampling_group.add_argument(
        "--generate",
        type=int,
        default=10,
        help="Number of new tokens to sample after training",
    )
    sampling_group.add_argument(
        "--no-newlines",
        action="store_true",
        help="During sampling/reporting, avoid emitting newline tokens",
    )
    sampling_group.add_argument(
        "--no-boundary",
        action="store_true",
        help="Allow completions to continue immediately after the prompt without enforcing a word boundary",
    )

    logging_group = parser.add_argument_group("Logging & diagnostics")
    logging_group.add_argument(
        "--time",
        action="store_true",
        help="Prefix training progress logs with local HH:MM timestamps",
    )
    logging_group.add_argument(
        "--debug-interrupt",
        action="store_true",
        help="If set, re-raise KeyboardInterrupt with a full stack trace.",
    )
    logging_group.add_argument(
        "--no-ansi",
        action="store_true",
        help="Suppress the parallel .ansi log (which preserves ANSI colors)",
    )
    logging_group.add_argument(
        "--no-escape-newline-tokens",
        action="store_true",
        help="Print an actual newline to the console and logfile if a completion or prompt contains a newline",
    )
    logging_group.add_argument(
        "--show-train-loss-details",
        action="store_true",
        help="Show the per-row loss columns in the live log",
    )
    logging_group.add_argument(
        "--no-show-test-loss-details",
        action="store_true",
        help="Collapse the test loss group down to a single column in the live log",
    )
    logging_group.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Exit after N seconds using a timer (0 disables the timeout)",
    )

    import_group = parser.add_argument_group("Checkpoint import/export")
    import_group.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Append an extra _TAG suffix to the model name (can be repeated)",
    )
    import_group.add_argument(
        "--pt",
        type=pathlib.Path,
        help="Load an explicit checkpoint file instead of the default model path",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="COMMAND",
        help="Action to perform",
    )
    parser.set_defaults(
        command=None,
        report_count=None,
        test_start=None,
    )


    # --------------------------------------------------------
    # Subcommand args parser for "train"

    train_parser = subparsers.add_parser(
        "train",
        help="Run the standard training loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.set_defaults(command="train")

    report_parser = subparsers.add_parser(
        "report",
        help="Skip training and generate completions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    report_parser.set_defaults(command="report")
    report_parser.add_argument(
        "-n",
        "--count",
        dest="report_count",
        type=int,
        default=10,
        help="How many completions to generate",
    )


    # --------------------------------------------------------
    # Subcommand args parser for "test"

    test_parser = subparsers.add_parser(
        "test",
        help="Print block-length tokens from the test corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    test_parser.set_defaults(command="test")
    test_parser.add_argument(
        "--start",
        dest="test_start",
        type=int,
        default=0,
        help="Cursor offset within the test corpus to begin printing",
    )


    # --------------------------------------------------------
    # Subcommand args parser for "profile"

    profile_parser = subparsers.add_parser(
        "profile",
        help="Run a warm-up and profiled training step, then dump profiler stats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    profile_parser.set_defaults(command="profile")


    # --------------------------------------------------------
    # Subcommand args parser for "size"

    size_parser = subparsers.add_parser(
        "size",
        help="Print parameter breakdown for the configured model and exit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    size_parser.set_defaults(command="size")
    size_parser.add_argument(
        "--check",
        dest="size_check",
        action="store_true",
        help="Instantiate the model and verify the analytic counts",
    )
    size_parser.add_argument(
        "--estimate",
        dest="size_estimate",
        action="store_true",
        help="Append a dominant-term estimate section",
    )


    # --------------------------------------------------------
    # Subcommand args parser for "prompts"

    prompt_parser = subparsers.add_parser(
        "prompts",
        help="Inspect or modify the prompt-tracking metadata stored in a checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prompt_parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Optional checkpoint path (defaults to the model path selected by global flags)",
    )
    prompt_parser.add_argument(
        "--list",
        action="store_true",
        help="List the stored prompts and exit",
    )
    prompt_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset prompts and statuses to the defaults hard-coded in grce.py",
    )
    prompt_parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove all stored prompts and reset statuses",
    )
    prompt_parser.add_argument(
        "--add",
        nargs=2,
        metavar=("PROMPT", "EXPECTED"),
        help="Append a new prompt/expected pair",
    )
    prompt_parser.add_argument(
        "--remove",
        type=int,
        action="append",
        default=[],
        help="Remove the prompt at index N (can be repeated)",
    )
    prompt_parser.set_defaults(command="prompts")


    # --------------------------------------------------------
    # Subcommand args parser for "create"

    create_parser = subparsers.add_parser(
        "create",
        help="Create a new checkpoint with random weights and exit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    create_parser.set_defaults(command="create")
    create_import_group = create_parser.add_argument_group("Checkpoint import tweaks")
    create_import_group.add_argument(
        "--import-model",
        dest="create_import_model",
        type=pathlib.Path,
        help="Initialize from another checkpoint when creating a new model",
    )
    create_import_group.add_argument(
        "--trim-model",
        dest="create_trim_model",
        action="store_true",
        help="Allow importing into a smaller model by dropping overflow",
    )
    create_import_group.add_argument(
        "--drop-layers",
        dest="create_drop_layers",
        type=str,
        default="",
        help="Comma-separated layer numbers (1-indexed) to remove during import",
    )
    create_import_group.add_argument(
        "--add-layers",
        dest="create_add_layers",
        type=str,
        default="",
        help="Comma-separated layer numbers (1-indexed) to insert during import",
    )

    corpus_parser = subparsers.add_parser(
        "corpus",
        help="List or configure corpora stored in a checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    corpus_parser.set_defaults(command="corpus")
    corpus_parser.add_argument(
        "--add",
        dest="corpus_add",
        metavar="NAME",
        type=str,
        help="Register a corpus without changing the current selection",
    )
    corpus_parser.add_argument(
        "--set",
        dest="corpus_set",
        metavar="NAME",
        type=str,
        help="Register a corpus (if needed) and make it current",
    )


    # --------------------------------------------------------
    # Run the args parser

    args = parser.parse_args()
    args.corpus = None

    if args.command is None:
        parser.print_help()
        parser.exit(
            1,
            "\nPlease specify a command (train, report, test, size, corpus, create, or prompts).\n",
        )


    # --------------------------------------------------------
    # Normalize, tweak, and check global options

    if args.tiny:
        if not flag_present("--vocab-size"):
            args.vocab_size = 600
        if not flag_present("--batch-size"):
            args.batch_size = 12
        if not flag_present("--block-size"):
            args.block_size = 6
        if not flag_present("--n-layer"):
            args.n_layer = 3
        if not flag_present("--n-head"):
            args.n_head = 2
        if not flag_present("--n-embd"):
            args.n_embd = 8
        if not flag_present("--n-grce"):
            args.n_grce = 4
        if not flag_present("--n-xctx"):
            args.n_xctx = 9
        if not flag_present("--steps"):
            args.steps = 2
        if not flag_present("--cycles"):
            args.cycles = 1
        if not flag_present("--eval-interval"):
            args.eval_interval = 1

    args._block_length_defined = args.block_length is not None
    if args.block_length is None:
        args.block_length = args.block_size
    if args.block_length <= 0:
        parser.error("--block-length must be positive")
    if args.block_length > args.block_size:
        parser.error("--block-length must be <= --block-size")

    args.prompt = normalize_prompt(args.prompt)

    # --------------------------------------------------------
    # Add non-inverted option names and values for --no-* options

    for k, v in list(args.__dict__.items()):
        if k.startswith("no_"):
            assert type(v) is bool
            args.__dict__[k[3:]] = not v
            # del args.__dict__[k]
    if not hasattr(args, "grad_summary"):
        args.grad_summary = getattr(args, "no_grad_summary", False) is False


    # --------------------------------------------------------
    # Parse "create" sub-command args

    if args.command == "create":
        def parse_layer_list(value: str, flag: str) -> list[int]:
            if not value:
                return []
            try:
                entries = [int(part) for part in value.split(",") if part]
            except ValueError as exc:
                raise ValueError(f"{flag} must be a comma-separated list of integers") from exc
            return entries

        args.create_args = Args(
            import_model=args.create_import_model,
            trim_model=args.create_trim_model,
            drop_layers=parse_layer_list(args.create_drop_layers, "--drop-layers"),
            add_layers=parse_layer_list(args.create_add_layers, "--add-layers"),
        )

        if (args.create_args.drop_layers or args.create_args.add_layers) and not args.create_args.import_model:
            raise ValueError("--drop-layers/--add-layers are only valid with --import-model")
        if args.create_args.trim_model and not args.create_args.import_model:
            raise ValueError("--trim-model is only valid with --import-model")

    return args


def args_to_model_geometry(args: Args):
    """Project the parsed CLI namespace into :class:`ModelGeometry` metadata.

    ``Runtime`` uses this helper when saving checkpoints or describing models
    so downstream tools and :func:`grce_main` can reload consistent geometry.
    """

    return ModelGeometry(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_grce=args.n_grce,
        n_xctx=args.n_xctx,
    )


if __name__ == "__main__":
    cli_args = grce_cli_args(sys.argv)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class Colors:
    """ANSI escape helpers consumed by :func:`color_text` and CLI logging."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    NOT_BOLD = "\033[22m"
    UNDERLINE = "\033[4m"
    NO_UNDERLINE = "\033[24m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    WHITE = "\033[97m"


def color_text(text: str, color: str, *, bold: bool = False, underline: bool = False) -> str:
    """Render text with ANSI styles for reporters such as :class:`Runtime`."""

    prefix = ""
    if bold:
        prefix += Colors.BOLD
    if underline:
        prefix += Colors.UNDERLINE
    return f"{prefix}{color}{text}{Colors.RESET}"



def prompt_needs_boundary(text: str) -> bool:
    """Return True when the prompt selection logic should enforce boundary tokens."""

    trimmed = text.rstrip()
    if not trimmed:
        return False
    return trimmed[-1].lower() in ASCII_LOWERCASE


def upgrade_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Upgrade legacy checkpoints that used older feed-forward key names."""

    needs_upgrade = any(".ff.net." in key for key in state)
    if not needs_upgrade:
        return state
    upgraded: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        marker = ".ff.net."
        if marker in key:
            prefix, suffix = key.split(marker, 1)
            if suffix.startswith("0."):
                new_key = f"{prefix}.ff.fc1.{suffix[2:]}"
            elif suffix.startswith("2."):
                new_key = f"{prefix}.ff.fc2.{suffix[2:]}"
            else:
                continue
        upgraded[new_key] = value
    return upgraded


class Tee:
    """Mirror stdout/stderr for :class:`Runtime` logging capture."""

    def __init__(self, *streams: tuple):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream, strip in self.streams:
            if strip:
                stream.write(ANSI_RE.sub("", data))
            else:
                stream.write(data)

    def flush(self) -> None:
        for stream, _ in self.streams:
            stream.flush()

import time
import threading
import pynvml

class GpuUtilSampler:
    """Background NVML sampler used by :func:`generate`/profiling utilities."""

    def __init__(self, device_index=0, interval_s=0.02):
        """
        interval_s: sampling period (20–50 ms is a good sweet spot)
        """
        self.interval_s = interval_s
        self.device_index = device_index

        self.busy_time_s = 0.0
        self.max_mem_util = 0.0

        self._running = False
        self._thread = None

    def start(self):
        assert not self._running
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

        self.busy_time_s = 0.0
        self.max_mem_util = 0.0

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        assert self._running
        self._running = False
        if self._thread:
            self._thread.join()
        pynvml.nvmlShutdown()

    def _run(self):
        last_t = time.time()
        while self._running:
            time.sleep(self.interval_s)
            t = time.time()
            dt = t - last_t
            last_t = t

            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            self.busy_time_s += (util.gpu / 100.0) * dt
            self.max_mem_util = max(self.max_mem_util, util.memory)

class Timer:
    """Wall/CPU/GPU timer aggregated by :class:`Runtime` for progress logs."""

    def __init__(self):
        self.gpu_sampler = GpuUtilSampler(interval_s=0.2)
        self.wall_secs = 0.0
        self.cpu_secs = 0.0
        self.gpu_secs = 0.0
        self.gpu_mem = 0.0
        self.wall_start = None
        self.cpu_start = None
        self.gpu_start = None
        self.gpu_sampler_active = False

    def add(self, other):
        assert self.wall_start is None
        assert other.wall_start is None
        self.wall_secs += other.wall_secs
        self.cpu_secs += other.cpu_secs
        self.gpu_secs += other.gpu_secs
        if self.gpu_mem is not None and other.gpu_mem is not None:
            self.gpu_mem = max(self.gpu_mem, other.gpu_mem)
        return self

    def sub(self, other):
        assert self.wall_start is None
        assert other.wall_start is None
        self.wall_secs = max(0.0, self.wall_secs - other.wall_secs)
        self.cpu_secs = max(0.0, self.cpu_secs - other.cpu_secs)
        self.gpu_secs = max(0.0, self.gpu_secs - other.gpu_secs)
        self.gpu_mem = None
        return self

    def ratio(self, other):
        return (f"{self.wall_secs/(other.wall_secs+0.001):.2f}x / "
                f"{self.cpu_secs/(other.cpu_secs+0.001):.2f}x / "
                f"{self.gpu_secs/(other.gpu_secs+0.001):.2f}x")

    def start(self):
        assert self.wall_start is None
        self.wall_start = time.time()
        self.cpu_start = time.process_time()
        self.gpu_sampler_active = False
        try:
            self.gpu_sampler.start()
            self.gpu_sampler_active = True
        except Exception:
            self.gpu_sampler_active = False
        return self

    def stop(self):
        assert self.wall_start is not None
        self.wall_secs += time.time() - self.wall_start
        self.cpu_secs += time.process_time() - self.cpu_start
        if self.gpu_sampler_active:
            try:
                self.gpu_sampler.stop()
                self.gpu_secs += self.gpu_sampler.busy_time_s
                if self.gpu_mem is not None:
                    self.gpu_mem = max(self.gpu_mem, self.gpu_sampler.max_mem_util)
            except Exception:
                pass
        self.gpu_sampler_active = False
        self.wall_start = None
        self.cpu_start = None
        return self

    def __str__(self):
        if self.gpu_mem is None or True:
            return f"{self.wall_secs:.2f}s / {self.cpu_secs:.2f}s / {self.gpu_secs:.2f}s"
        return f"{self.wall_secs:.2f}s / {self.cpu_secs:.2f}s / {self.gpu_secs:.2f}s / {self.gpu_mem:.2f}%"


# -----------------------------------------------------------------------------
# GRCE Model Size Information
# -----------------------------------------------------------------------------

def _get_inner_xctx_width(config: GeometryLike) -> int:
    """Return the XCTX inner width used by :class:`TransformerXCTX` samplers."""

    return min(
        config.n_embd // 2,
        config.n_xctx // 2,
        max(config.n_embd // 4, config.n_xctx // config.n_layer),
    )


def _build_geometry(config: GeometryLike, block_size: int) -> list[tuple[str, str, int]]:
    """Assemble the (label, description, value) tuples for ``size`` reports."""

    return [
        ("V", "vocab size", config.vocab_size),
        ("B", "block size", block_size),
        ("L", "transform layers", config.n_layer),
        ("H", "attention heads", config.n_head),
        ("E", "embedding width", config.n_embd),
        ("G", "grce width", config.n_grce),
        ("X", "xctx width", config.n_xctx),
        ("U", "inner xctx width", _get_inner_xctx_width(config)),
    ]

def _expected_sections(config: GeometryLike, block_size: int) -> list[tuple[str, str, list[dict]]]:
    """Return analytic section breakdown consumed by :func:`grce_cmd_size`."""

    V = config.vocab_size
    B = block_size
    L = config.n_layer
    H = config.n_head
    E = config.n_embd
    G = config.n_grce
    X = config.n_xctx
    sections: list[tuple[str, str, list[dict]]] = []

    def eval_items(items):
        for item in items:
            item["count"] = eval(item["formula"], {
                "V": config.vocab_size,
                "B": block_size,
                "L": config.n_layer,
                "H": config.n_head,
                "E": config.n_embd,
                "G": config.n_grce,
                "X": config.n_xctx
            })
        return items

    global_items = eval_items([
        {"label": "token embeddings", "formula": "V * E"},
        {"label": "position embeddings", "formula": "B * E"},
        {"label": "special embeddings", "formula": "0 * E"},
        {"label": "grce embeddings", "formula": "0 * G"},
    ])
    sections.append(("embeddings", "Embeddings", global_items))

    transformer_items = eval_items([
        {
            "label": "attn qkv",
            "formula": "3 * L * (E*E + E)",
        },
        {
            "label": "attn proj",
            "formula": "L * (E*E + E)",
        },
        {
            "label": "ffn fc1",
            "formula": "L * (4*E*E + 4*E)",
        },
        {
            "label": "ffn fc2",
            "formula": "L * (4*E*E + E)",
        },
    ])
    sections.append(("transformer", "Transformer", transformer_items))

    if G > 0:
        grce_items = eval_items([
            {
                "label": "samplers",
                "formula": "L * (2*E + E*G + G)",
            },
            {
                "label": "mlp",
                "formula": "8*G*G + 9*G",
            },
            {
                "label": "bias",
                "formula": "L * (G*E + E)",
            },
        ])
    else:
        grce_items = []
    sections.append(("grce", "GRCE Channel", grce_items))

    if X > 0:
        chunk = X // max(1, config.n_layer)
        mid = max(1, (4 * X) // max(1, config.n_layer))
        xctx_items = [
            {
                "label": "samplers",
                "count": config.n_layer
                * (2 * E + E * chunk + chunk + chunk * X + X),
                "formula": "L * (2*E + E*(X/L) + (X/L) + (X/L)*X + X)",
            },
            {
                "label": "mlp",
                "count": 2 * X + (X * mid + mid) + (mid * X + X) + 2 * X,
                "formula": "2*X + (X*M + M) + (M*X + X) + 2*X (M=4*X/L)",
            },
            {
                "label": "bias",
                "count": max(0, config.n_layer) * (X * chunk + chunk + chunk * E + E),
                "formula": "L * (X*(X/L) + (X/L) + (X/L)*E + E)",
            },
        ]
    else:
        xctx_items = []
    sections.append(("xctx", "XCTX Channel", xctx_items))

    # Placeholder for summary, filled later
    return sections


def _append_summary_section(
    sections: list[tuple[str, str, list[dict]]],
) -> list[tuple[str, str, list[dict]]]:
    """Add the total row used in :func:`grce_cmd_size` output."""

    totals: dict[str, int] = {}
    for key, _title, items in sections:
        totals[key] = sum(item["count"] for item in items)
    summary_items = [
        {"label": "embeddings", "count": totals.get("embeddings", 0), "formula": ""},
        {
            "label": "transformer",
            "count": totals.get("transformer", 0),
            "formula": "",
        },
        {"label": "grce channel", "count": totals.get("grce", 0), "formula": ""},
        {"label": "xctx channel", "count": totals.get("xctx", 0), "formula": ""},
    ]
    overall = sum(item["count"] for item in summary_items)
    summary_items.append({"label": "total", "count": overall, "formula": ""})
    sections.append(("summary", "Model Parameter Breakdown", summary_items))
    return sections


def _print_geometry(geometry: list[tuple[str, str, int]]) -> None:
    """Render the geometry table for :func:`grce_cmd_size`."""

    print(color_text("Model Geometry", Colors.CYAN, bold=True))
    for var, desc, value in geometry:
        print(f"  {var} ({desc:<17s}): {value}")


def _print_section(title: str, items: list[dict]) -> int:
    """Print a single section inside :func:`grce_cmd_size`."""

    print(color_text(title, Colors.CYAN, bold=True))
    if not items:
        print("  disabled")
        return 0
    total = 0
    for entry in items:
        label = entry["label"]
        count = entry["count"]
        formula = entry.get("formula", "")
        total += count
        line = f"  {label:<20} {count:>15,}"
        if formula:
            line += f"  ({formula})"
        print(line)
    print(f"  {'total':<20} {total:>15,}")
    return total


def _flatten_expected(
    sections: list[tuple[str, str, list[dict]]],
) -> dict[tuple[str, str], int]:
    """Map ``(section, label)`` keys to analytic counts for size checking."""

    mapping: dict[tuple[str, str], int] = {}
    for key, _title, items in sections:
        if key == "summary":
            continue
        for entry in items:
            mapping[(key, entry["label"])] = entry["count"]
    return mapping


def _compute_actual_counts(config: GeometryLike) -> dict[tuple[str, str], int]:
    """Instantiate :class:`GRCEGPT` to validate :func:`grce_cmd_size` numbers."""

    model = GRCEGPT(config)
    counts: dict[tuple[str, str], int] = {}

    counts[("embeddings", "token embeddings")] = _module_param_count(model.core.tok_emb)
    counts[("embeddings", "position embeddings")] = _module_param_count(model.core.pos_emb)

    attn_qkv = 0
    attn_proj = 0
    ffn_fc1 = 0
    ffn_fc2 = 0
    for block in model.core.blocks:
        attn_qkv += sum(
            _module_param_count(getattr(block.attn, attr))
            for attr in ("key", "query", "value")
        )
        attn_proj += _module_param_count(block.attn.proj)
        ffn_fc1 += _module_param_count(block.ff.fc1)
        ffn_fc2 += _module_param_count(block.ff.fc2)
    counts[("transformer", "attn qkv")] = attn_qkv
    counts[("transformer", "attn proj")] = attn_proj
    counts[("transformer", "ffn fc1")] = ffn_fc1
    counts[("transformer", "ffn fc2")] = ffn_fc2

    for channel in model.context_channels:
        if channel.disabled:
            continue
        key = "xctx" if type(channel) is TransformerXCTX else "grce"
        breakdown = channel.parameter_breakdown()
        for label, value in breakdown.items():
            counts[(key, label)] = counts.get((key, label), 0) + value
    return counts


def _dominant_estimates(config: GeometryLike) -> list[tuple[str, int, str]]:
    """Return asymptotic parameter counts shown by ``size --estimate``."""

    L = config.n_layer
    E = config.n_embd
    G = config.n_grce
    X = config.n_xctx
    U = _get_inner_xctx_width(config)
    estimates = [
        ("transformer", 12*L*E*E, "(12*L*E^2)"),
    ]
    if G > 0:
        estimates.append(
            (
                "grce channel",
                2*L*E*G + 8*G*G,
                "(2*L*E*G + 8*G^2)",
            )
        )
    if X > 0:
        estimates.append(
            (
                "xctx channel",
                2*E*U + 2*X*U + 2*X*X,
                "(2*E*U + 2*X*U + 2*X^2)",
            )
        )
    overall = sum(item[1] for item in estimates)
    estimates.append(("total", overall, ""))
    return estimates


def grce_cmd_size(
    args: GeometryLike,
    *,
    check: bool = False,
    estimate: bool = False,
) -> None:
    """Emit the ``size`` subcommand report.

    Print the standard parameter breakdown based on :class:`ModelGeometry`.
    Called exclusively from :func:`grce_main`.
    """
    print()
    geometry = _build_geometry(args, args.block_size)
    sections = _append_summary_section(_expected_sections(args, args.block_size))
    _print_geometry(geometry)
    for idx, (key, title, items) in enumerate(sections):
        print()
        if key == "summary":
            print(color_text(title, Colors.CYAN, bold=True))
            for entry in items:
                label = entry["label"]
                count = entry["count"]
                line = f"  {label:<20} {count:>15,}"
                print(line)
            continue
        _print_section(title, items)

    if estimate:
        print()
        print(color_text("Estimate using dominant terms only (excl. embeddings)", Colors.CYAN, bold=True))
        for label, count, formula in _dominant_estimates(args):
            print(f"  {label:<20} {count:>15,}  {formula}")

    if check:
        expected_map = _flatten_expected(sections)
        actual_map = _compute_actual_counts(args)
        mismatches: list[tuple[str, str, int, int]] = []
        for (key, label), expected in expected_map.items():
            actual = actual_map.get((key, label), 0)
            if expected != actual:
                mismatches.append((key, label, expected, actual))
        if mismatches:
            print(color_text("\n[size --check] mismatches detected:", Colors.RED, bold=True))
            for key, label, expected, actual in mismatches:
                print(
                    f"  {key}:{label} expected {expected:,} but model has {actual:,}"
                )
        else:
            print(
                color_text(
                    "\n[size --check] analytic counts match instantiated model", Colors.GREEN
                )
            )

    print()
    return 0

def grce_cli_size(args: Args):
    """Thin wrapper invoked by ``grce.py size`` before torch imports."""

    assert args.command == "size"
    return grce_cmd_size(args,
        check=args.size_check,
        estimate=args.size_estimate
    )

if __name__ == "__main__":
    # run it here when not in --check mode, and run it later if we need torch for --check
    # on some builds it can take 5 seconds or longer to import torch, so we early-exit
    # on the "size" sub-command here so it prints the model size right away without that delay
    if cli_args.command == "size" and not cli_args.size_check:
        sys.exit(grce_cli_size(cli_args))


# -----------------------------------------------------------------------------
# GRCE Library Components
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import string
from tokenizers import Tokenizer

ASCII_LETTERS = set(string.ascii_letters)
ASCII_LOWERCASE = set(string.ascii_lowercase)
class RMSNorm(nn.Module):
    """Root-mean-square norm used by the XCTX recurrent path."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(rms + self.eps)
        return x * scale * self.weight


class LayerDampening(nn.Module):
    """
    LD: Layer-Dampening

    - Removes mean (LN-style)
    - Applies per-feature learned gain
    - Uses soft radial dampening instead of hard RMS normalization

    For input x [..., d]:
        r = ||x||
        denom = 1 + softplus(k * (r - 1)) / k
        y = gain * x / denom
    """

    def __init__(self, dim, with_gain=True, eps=1e-8, init_log_k=0.0):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim)) if with_gain else None
        self.log_k = nn.Parameter(torch.tensor(init_log_k))

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        r = torch.linalg.norm(x, dim=-1, keepdim=True)
        r = r.clamp_min(self.eps) # just to be on the safe side
        k = torch.exp(self.log_k)
        denom = 1.0 + F.softplus(k * (r - 1.0)) / k
        y = x / denom
        if self.gain is None:
            return y
        return y * self.gain


def default_prompt_entries() -> list[tuple[str, str]]:
    """Return the built-in prompt catalog."""

    return [(prompt, expected) for prompt, expected in PROMPT_GOALS]


@dataclass
class PromptEntry:
    expected: str
    maxarg_count: int = 0
    sample_count: int = 0


class PromptRegistry:
    """Maintains prompt metadata for training diagnostics."""

    def __init__(self, tokenizer: GPT2TokenizerWrapper, data: dict | None = None) -> None:
        self.tokenizer = tokenizer
        self.entries: OrderedDict[str, PromptEntry] = OrderedDict()
        self._expected_cache: dict[str, list[int]] = {}
        if data:
            self._load_from_data(data)
        else:
            self.reset_to_defaults()

    def _load_from_data(self, data: dict) -> None:
        self.entries.clear()
        self._expected_cache.clear()
        for prompt, payload in data.items():
            if not isinstance(prompt, str) or not isinstance(payload, dict):
                continue
            expected = str(payload.get("expected", ""))
            maxarg = int(payload.get("maxarg_count", 0) or 0)
            sample = int(payload.get("sample_count", 0) or 0)
            self.entries[prompt] = PromptEntry(expected, maxarg, sample)
        if not self.entries:
            self.reset_to_defaults()

    def reset_to_defaults(self) -> None:
        self.entries.clear()
        for prompt, expected in PROMPT_GOALS:
            self.entries[prompt] = PromptEntry(expected)
        self._expected_cache.clear()

    def clear(self) -> None:
        self.entries.clear()
        self._expected_cache.clear()

    def add_prompt(self, prompt: str, expected: str) -> None:
        self.entries[prompt] = PromptEntry(expected)
        self._expected_cache.pop(prompt, None)

    def remove_indices(self, indices: list[int]) -> None:
        prompts = list(self.entries.keys())
        for idx in sorted(set(indices), reverse=True):
            if 0 <= idx < len(prompts):
                key = prompts[idx]
                self.entries.pop(key, None)
                self._expected_cache.pop(key, None)

    def serialize(self) -> dict[str, dict[str, object]]:
        return {
            prompt: {
                "expected": entry.expected,
                "maxarg_count": int(entry.maxarg_count),
                "sample_count": int(entry.sample_count),
            }
            for prompt, entry in self.entries.items()
        }

    def ordered_entries(self) -> list[tuple[str, PromptEntry]]:
        return list(self.entries.items())

    def pick_prompt(self, mode: str, default_prompt: str) -> tuple[str, str | None, bool]:
        entries = self.ordered_entries()
        weights: list[float] = []
        for _, entry in entries:
            count = entry.maxarg_count if mode == "argmax" else entry.sample_count
            weights.append(1.0 / (1.0 + count))
        total_weight = sum(weights) + 1.0
        choice = random.random() * total_weight
        if choice < 1.0:
            return default_prompt, None, True
        choice -= 1.0
        for (prompt, entry), weight in zip(entries, weights):
            choice -= weight
            if choice <= 0:
                return prompt, entry.expected, False
        if entries:
            prompt, entry = entries[-1]
            return prompt, entry.expected, False
        return default_prompt, None, True

    def _expected_ids(self, prompt: str, expected: str) -> list[int]:
        cached = self._expected_cache.get(prompt)
        if cached is not None:
            return cached
        ids = self.tokenizer.encode_ids(expected)
        self._expected_cache[prompt] = ids
        return ids

    def record_result(
        self,
        prompt: str,
        expected: str,
        *,
        used_argmax: bool,
        completion_ids: list[int],
    ) -> bool:
        entry = self.entries.get(prompt)
        if entry is None or not expected:
            return False
        expected_ids = self._expected_ids(prompt, expected)
        if not expected_ids or len(completion_ids) < len(expected_ids):
            return False
        if completion_ids[: len(expected_ids)] != expected_ids:
            return False
        if used_argmax:
            entry.maxarg_count += 1
        else:
            entry.sample_count += 1
        return True


class GPT2TokenizerWrapper:
    """Minimal tokenizer shim for prompt encoding/decoding."""

    def __init__(
        self,
        *,
        tokenizer_path: pathlib.Path | None = None,
        tokenizer_json: str | None = None,
    ) -> None:
        if tokenizer_json is not None:
            self._tokenizer = Tokenizer.from_str(tokenizer_json)
        elif tokenizer_path is not None:
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            raise ValueError("Tokenizer path or JSON must be provided")
        self.vocab_size = self._tokenizer.get_vocab_size()
        self.leading_alpha_token_ids = sorted(self._collect_leading_alpha_tokens())

    def _collect_leading_alpha_tokens(self) -> set[int]:
        vocab = self._tokenizer.get_vocab()
        token_ids: set[int] = set()
        for token, tok_id in vocab.items():
            piece = self._tokenizer.decode([tok_id], skip_special_tokens=False)
            if piece and piece[0] in ASCII_LOWERCASE:
                token_ids.add(tok_id)
        return token_ids

    def encode(self, text: str) -> torch.Tensor:
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        return torch.tensor(ids, dtype=torch.long)

    def encode_ids(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: torch.Tensor | list[int]) -> str:
        if isinstance(tokens, torch.Tensor):
            ids = tokens.tolist()
        else:
            ids = list(tokens)
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def decode_one(self, token: int) -> str:
        return self._tokenizer.decode([token], skip_special_tokens=False)

    def decode_pretty(
        self,
        args: Args,
        tokens: torch.Tensor,
        color: str = Colors.MAGENTA,
        altcolor: str = Colors.GREEN,
        alt: bool = False,
    ) -> str:
        if alt:
            color, altcolor = Colors.YELLOW, Colors.CYAN
        parts: list[str] = []
        for tok in tokens.tolist():
            s = self.decode_one(tok)
            if not s:
                continue
            if s == " " or " " in s[1:]:
                s = s.replace(" ", FANCY_SPACE)
            replacement = (
                FANCY_ENTER if args.escape_newline_tokens else FANCY_ENTER.replace(" ", "\n")
            )
            s = s.replace("\n", replacement)
            parts.append(color + s + Colors.RESET)
            color, altcolor = altcolor, color
        return "".join(parts)


def load_cached_tokens(split: str, cache_path: pathlib.Path) -> torch.Tensor:
    """Load tokens produced by corpus.py tokens."""

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Token cache {cache_path} not found for {split}; run corpus.py tokens to build it."
        )
    payload = torch.load(cache_path, map_location="cpu")
    tokens = payload.get("tokens")
    if tokens is None:
        raise ValueError(f"Token cache {cache_path} is missing 'tokens' data")
    return tokens.long()


@dataclass
class TextDataset:
    """Holds rolling corpus state for :class:`Runtime` training/eval loops."""

    train_tokens: torch.Tensor
    test_tokens: torch.Tensor
    train_text: str | None
    test_text: str | None
    train_path: pathlib.Path
    test_path: pathlib.Path
    positions: Dict[str, int] = field(
        default_factory=lambda: {"train": 0, "test": 0}
    )
    cycles: Dict[str, int] = field(
        default_factory=lambda: {"train": 0, "test": 0}
    )
    chunks: Dict[str, torch.Tensor] = field(default_factory=dict)
    chunk_offsets: Dict[str, int] = field(default_factory=dict)

    def state_dict(self) -> Dict[str, int]:
        return {
            "train_cursor": int(self.positions.get("train", 0)),
            "test_cursor": int(self.positions.get("test", 0)),
            "train_cycles": int(self.cycles.get("train", 0)),
            "test_cycles": int(self.cycles.get("test", 0)),
            "train_count": int(self.train_tokens.numel()),
            "test_count": int(self.test_tokens.numel()),
        }

    def load_state(self, state: dict | None) -> None:
        self.positions = {"train": 0, "test": 0}
        self.cycles = {"train": 0, "test": 0}
        if not state:
            return
        if "positions" in state:
            legacy_positions = state.get("positions", {})
            for split in ("train", "test"):
                value = int(legacy_positions.get(split, 0) or 0)
                total = len(self.train_tokens if split == "train" else self.test_tokens)
                if total:
                    value %= total
                self.positions[split] = value
            return
        for split, key in (("train", "train_cursor"), ("test", "test_cursor")):
            value = int(state.get(key, 0) or 0)
            total = len(self.train_tokens if split == "train" else self.test_tokens)
            if total:
                value %= total
            self.positions[split] = value
        for split, key in (("train", "train_cycles"), ("test", "test_cycles")):
            value = int(state.get(key, 0) or 0)
            self.cycles[split] = max(0, value)

    def prepare_cycle(self, split: str, total_chars: int) -> bool:
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split {split!r}")
        source = self.train_tokens if split == "train" else self.test_tokens
        text = self.train_text if split == "train" else self.test_text
        if total_chars <= 0 or total_chars > len(source):
            total_chars = len(source)
        start = self.positions[split]
        chunk, _ = self._slice_with_wrap(source, text, start, total_chars)
        if len(chunk) <= 1:
            raise ValueError(f"Not enough tokens in {split} split to build a chunk")
        wrapped = False
        if len(source) > 0:
            span = start + total_chars
            wrapped = span >= len(source)
            self.positions[split] = span % len(source)
            if wrapped:
                self.cycles[split] = self.cycles.get(split, 0) + 1
        self.chunks[split] = chunk
        self.chunk_offsets[split] = start % len(source)
        return wrapped

    def get_batch(
        self,
        split: str,
        block_length: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks.get(split)
        if chunk is None:
            raise RuntimeError(
                f"No cached chunk for split {split}. Call prepare_cycle first."
            )
        chunk_offset = self.chunk_offsets.get(split)
        if chunk_offset is None:
            raise RuntimeError(f"Missing chunk offset for split {split}")
        span = block_length + 1
        if len(chunk) <= span:
            raise ValueError(
                f"Chunk for {split} must be larger than block length ({len(chunk)} <= {span})."
            )
        max_start = len(chunk) - span
        ix = torch.randint(0, max_start + 1, (batch_size,))
        windows = [chunk[i : i + span] for i in ix]
        stacked = torch.stack(windows)
        x = stacked[:, :-1].contiguous().to(device)
        y = stacked[:, 1:].contiguous().to(device)
        return x, y

    def _slice_with_wrap(
        self,
        tokens: torch.Tensor,
        text: str,
        start: int,
        needed: int,
    ) -> Tuple[torch.Tensor, list[str]]:
        n = len(tokens)
        first_take = min(needed, n - start)
        second_take = needed - first_take
        parts = []
        texts: list[str] = []
        if first_take:
            parts.append(tokens[start : start + first_take])
            if text is not None:
                texts.append(text[start : start + first_take])
        if second_take:
            parts.append(tokens[:second_take])
            if text is not None:
                texts.append(text[:second_take])
        chunk = torch.cat(parts) if len(parts) > 1 else parts[0]
        return chunk.contiguous(), texts

    def looped_slice(
        self,
        split: str,
        start: int,
        length: int,
    ) -> torch.Tensor:
        if length <= 0:
            return torch.empty(0, dtype=self.train_tokens.dtype)
        tokens = self.train_tokens if split == "train" else self.test_tokens
        total = len(tokens)
        if total == 0:
            raise ValueError(f"No tokens available for split {split!r}")
        start = start % total
        remaining = length
        pieces: list[torch.Tensor] = []
        pos = start
        while remaining > 0:
            take = min(remaining, total - pos)
            if take == 0:
                pos = 0
                continue
            pieces.append(tokens[pos : pos + take])
            remaining -= take
            pos = (pos + take) % total
        return torch.cat(pieces).contiguous()

    def sample_window(
        self,
        split: str,
        span: int,
        *,
        rng: random.Random | None = None,
    ) -> "TokenWindow":
        if span <= 0:
            raise ValueError("Window span must be positive")
        chunk = self.chunks.get(split)
        if chunk is None:
            raise RuntimeError(f"No cached chunk for split {split!r}; call prepare_cycle first")
        total = int(chunk.size(0))
        if total <= 0:
            raise ValueError(f"Chunk for split {split!r} is empty")
        if span > total:
            raise ValueError(
                f"Requested window ({span}) exceeds chunk length ({total}) for split {split}"
            )
        rng = rng or random
        max_offset = total - span
        offset = rng.randint(0, max_offset) if max_offset > 0 else 0
        return TokenWindow(chunk=chunk, start=offset, length=span, rng=rng)


@dataclass
class TokenWindow:
    chunk: torch.Tensor
    start: int
    length: int
    rng: random.Random

    def subwindow(self, span: int) -> "TokenWindow":
        if span <= 0:
            raise ValueError("Subwindow span must be positive")
        if span > self.length:
            raise ValueError(
                f"Requested subwindow ({span}) exceeds parent length ({self.length})"
            )
        max_offset = self.length - span
        offset = self.rng.randint(0, max_offset) if max_offset > 0 else 0
        return TokenWindow(
            chunk=self.chunk,
            start=self.start + offset,
            length=span,
            rng=self.rng,
        )

    def sample_batch(
        self,
        block_length: int,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive when sampling tokens")
        span = block_length + 1
        if span <= 1:
            raise ValueError("Block length must be >= 1 when sampling tokens")
        if span > self.length:
            raise ValueError(
                f"Sequence span ({span}) exceeds available window ({self.length})"
            )
        max_offset = self.length - span
        if max_offset > 0:
            offsets = torch.randint(0, max_offset + 1, (batch_size,))
        else:
            offsets = torch.zeros((batch_size,), dtype=torch.long)
        offsets = offsets.tolist()
        windows = [
            self.chunk[self.start + offset : self.start + offset + span]
            for offset in offsets
        ]
        stacked = torch.stack(windows)
        x = stacked[:, :-1].contiguous().to(device=device, dtype=torch.long)
        y = stacked[:, 1:].contiguous().to(device=device, dtype=torch.long)
        return x, y


# -----------------------------------------------------------------------------
# GRCE Model Components
# -----------------------------------------------------------------------------


def _module_param_count(module: nn.Module) -> int:
    """Return ``sum(p.numel())`` for :func:`grce_cmd_size` sanity checks."""

    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _module_list_param_count(modules: nn.ModuleList) -> int:
    """Aggregate :func:`_module_param_count` across ``ModuleList`` members."""

    return sum(_module_param_count(m) for m in modules)


class CausalSelfAttention(nn.Module):
    """GPT-style attention block used inside :class:`Block`."""

    def __init__(self, args: Args) -> None:
        super().__init__()
        config = args
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        dropout_positions: torch.Tensor | None = None,
        disable_rows: torch.Tensor | None = None,
        full_attention: bool = False,
        kv_cache_sources: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
        qh_query_callback=None,
        attn_mode: str = "decode",
        layer_idx: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        head_dim = C // self.n_head
        k_full = self.key(x)
        q = self.query(x).view(B, T, self.n_head, head_dim).transpose(1, 2)
        v_full = self.value(x)
        k_local = k_full.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v_local = v_full.view(B, T, self.n_head, head_dim).transpose(1, 2)

        cache_keys: list[torch.Tensor] = []
        cache_values: list[torch.Tensor] = []
        if kv_cache_sources:
            for pair in kv_cache_sources:
                if pair is None:
                    continue
                k_cache, v_cache = pair
                if k_cache is None or v_cache is None:
                    continue
                if k_cache.dim() != 4 or v_cache.dim() != 4:
                    raise ValueError("KV cache tensors must have rank 4 [rows, len, heads, head_dim]")
                cache_keys.append(k_cache.permute(0, 2, 1, 3))
                cache_values.append(v_cache.permute(0, 2, 1, 3))
        if cache_keys:
            cat_keys = torch.cat(cache_keys, dim=2)
            cat_values = torch.cat(cache_values, dim=2)
            all_k = torch.cat([cat_keys, k_local], dim=2)
            all_v = torch.cat([cat_values, v_local], dim=2)
            cache_len = cat_keys.size(2)
        else:
            all_k = k_local
            all_v = v_local
            cache_len = 0

        total_len = all_k.size(2)
        scores = (q @ all_k.transpose(-2, -1)) / math.sqrt(head_dim)

        block_mask: torch.Tensor | None = None
        if attn_mode not in {"encode", "decode", "noattn"}:
            raise ValueError(f"Unknown attention mode: {attn_mode}")
        if attn_mode == "decode" and not full_attention:
            block_mask = self.tril[:T, :T] == 0
        elif attn_mode == "noattn":
            eye = torch.eye(T, dtype=torch.bool, device=x.device)
            block_mask = ~eye
        if block_mask is not None:
            if cache_len > 0:
                prefix = torch.zeros(T, cache_len, dtype=torch.bool, device=x.device)
                combined_mask = torch.cat([prefix, block_mask], dim=1)
            else:
                combined_mask = block_mask
            scores = scores.masked_fill(combined_mask.bool()[None, None, :, :], float("-inf"))

        if dropout_positions is not None:
            valid = (dropout_positions >= 0).nonzero(as_tuple=False).flatten()
            if valid.numel() > 0:
                atten_block_mask = torch.zeros(B, T, total_len, dtype=torch.bool, device=x.device)
                for b_idx in valid.tolist():
                    pos = int(dropout_positions[b_idx].item())
                    if 0 <= pos < T:
                        target_col = cache_len + pos
                        if target_col < total_len and pos + 1 < T:
                            atten_block_mask[b_idx, pos + 1 :, target_col] = True
                scores = scores.masked_fill(atten_block_mask[:, None, :, :], float("-inf"))

        kv_output = (
            k_full.view(B, T, self.n_head, head_dim),
            v_full.view(B, T, self.n_head, head_dim),
        )

        local_max = scores.max(dim=-1).values
        exp_scores = torch.exp(scores - local_max.unsqueeze(-1))
        local_sum = exp_scores.sum(dim=-1)
        local_sum = torch.clamp(local_sum, min=1e-9)

        qh_data = None
        if qh_query_callback is not None:
            q_for_callback = q.transpose(1, 2)
            callback_result = qh_query_callback(q_for_callback, layer_idx)
            if callback_result is not None:
                qh_M, qh_S, qh_T = callback_result
                if qh_M is not None and qh_S is not None and qh_T is not None:
                    ext_M = qh_M.permute(0, 2, 1)
                    ext_S = qh_S.permute(0, 2, 1)
                    ext_T = qh_T.permute(0, 2, 1, 3)
                    qh_data = (ext_M, ext_S, ext_T)

        if qh_data is not None:
            ext_M, ext_S, ext_T = qh_data
            base_max = torch.maximum(local_max, ext_M)
            local_scale = torch.exp(local_max - base_max)
            ext_scale = torch.exp(ext_M - base_max)
            denom = local_scale * local_sum + ext_scale * ext_S
            denom = torch.clamp(denom, min=1e-9)
            local_weights = local_scale.unsqueeze(-1) * exp_scores / denom.unsqueeze(-1)
            ext_coeff = (ext_scale * ext_S) / denom
        else:
            local_weights = exp_scores / local_sum.unsqueeze(-1)
            ext_coeff = None

        local_weights = self.dropout(local_weights)
        y_local = torch.einsum("bhtl,bhlv->bhtv", local_weights, all_v)
        attn_output = y_local
        if ext_coeff is not None:
            attn_output = attn_output + ext_coeff.unsqueeze(-1) * ext_T

        y = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        if disable_rows is not None and disable_rows.any():
            row_mask = (~disable_rows).view(-1, 1, 1).to(y.dtype)
            y = y * row_mask
        return self.proj(y), kv_output

    def forward_incremental(
        self,
        x: torch.Tensor,
        cache: "LayerCache",
        *,
        puncture_mask: torch.Tensor | None = None,
        disable_rows: torch.Tensor | None = None,
        write_cache: bool = True,
    ) -> tuple[torch.Tensor, "LayerCache"]:
        if x.size(1) != 1:
            raise ValueError("Incremental attention expects a single-token sequence")
        B, T, C = x.shape
        k_full = self.key(x)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v_full = self.value(x)
        v = v_full.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k_new = k_full.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        key_append = k_new.squeeze(2).unsqueeze(2)
        value_append = v.squeeze(2).unsqueeze(2)
        if write_cache:
            cache.append(key_append, value_append)
            k, v = cache.tensors()
        else:
            k, v = key_append, value_append
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if puncture_mask is not None:
            att = att.masked_fill(puncture_mask[:, None, None, :], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        if disable_rows is not None and disable_rows.any():
            row_mask = (~disable_rows).view(-1, 1, 1).to(y.dtype)
            y = y * row_mask
        return self.proj(y), cache


class FeedForward(nn.Module):
    """Position-wise MLP reused by every :class:`Block`."""

    def __init__(self, args: Args) -> None:
        super().__init__()
        config = args
        hidden = 4 * config.n_embd
        self.fc1 = nn.Linear(config.n_embd, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, *, record_mask: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.fc1(x)
        mask = None
        if record_mask:
            mask = hidden[:, -1, :] > 0
        activated = self.act(hidden)
        out = self.fc2(activated)
        out = self.drop(out)
        return out, mask


class Block(nn.Module):
    """Transformer block consumed by :class:`TransformerStackCore` and :class:`GPTCore`."""

    def __init__(self, args: Args) -> None:
        super().__init__()
        config = args
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)
        self.xctx_attn_gain = nn.Parameter(torch.ones(config.n_embd))
        self.xctx_mlp_gain = nn.Parameter(torch.ones(config.n_embd))
        self.grce_attn_ld = LayerDampening(config.n_embd)
        self.grce_mlp_ld = LayerDampening(config.n_embd)

    def _apply_xctx_bias(
        self, tensor: torch.Tensor, bias: torch.Tensor | None, gain: torch.Tensor
    ) -> torch.Tensor:
        if bias is None:
            return tensor
        scaled = bias * gain.view(1, 1, -1)
        return tensor + scaled

    def _apply_grce_bias(
        self, tensor: torch.Tensor, bias: torch.Tensor | None, ld_layer: LayerDampening
    ) -> torch.Tensor:
        if bias is None:
            return tensor
        return tensor + ld_layer(bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        xctx_bias: torch.Tensor | None = None,
        grce_bias: torch.Tensor | None = None,
        kv_cache_sources: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
        qh_query_callback=None,
        attn_mode: str = "decode",
        layer_idx: int = 0,
        record_mask: bool = False,
        attention_disabled_rows: torch.Tensor | None = None,
        attention_dropout_positions: torch.Tensor | None = None,
        full_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_input = self._apply_xctx_bias(x, xctx_bias, self.xctx_attn_gain)
        attn_norm = self.ln1(attn_input)
        attn_norm = self._apply_grce_bias(attn_norm, grce_bias, self.grce_attn_ld)
        attn_output, kv_pair = self.attn(
            attn_norm,
            disable_rows=attention_disabled_rows,
            dropout_positions=attention_dropout_positions,
            full_attention=full_attention,
            kv_cache_sources=kv_cache_sources,
            qh_query_callback=qh_query_callback,
            attn_mode=attn_mode,
            layer_idx=layer_idx,
        )
        if attention_disabled_rows is not None and attention_disabled_rows.any():
            mask = (~attention_disabled_rows).view(-1, 1, 1).to(attn_output.dtype)
            attn_output = attn_output * mask
        x = x + attn_output
        ff_input = self._apply_xctx_bias(x, xctx_bias, self.xctx_mlp_gain)
        pre_ff = self.ln2(ff_input)
        pre_ff = self._apply_grce_bias(pre_ff, grce_bias, self.grce_mlp_ld)
        ff_out, mask = self.ff(pre_ff, record_mask=record_mask)
        x = x + ff_out
        return x, mask, kv_pair

    def forward_incremental(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        *,
        xctx_bias: torch.Tensor | None = None,
        grce_bias: torch.Tensor | None = None,
        record_mask: bool = False,
        attention_disabled_rows: torch.Tensor | None = None,
        puncture_mask: torch.Tensor | None = None,
        write_cache: bool = True,
    ) -> tuple[torch.Tensor, LayerCache, torch.Tensor | None]:
        attn_input = self._apply_xctx_bias(x, xctx_bias, self.xctx_attn_gain)
        attn_norm = self.ln1(attn_input)
        attn_norm = self._apply_grce_bias(attn_norm, grce_bias, self.grce_attn_ld)
        attn_out, cache = self.attn.forward_incremental(
            attn_norm,
            cache,
            puncture_mask=puncture_mask,
            disable_rows=attention_disabled_rows,
            write_cache=write_cache,
        )
        if attention_disabled_rows is not None and attention_disabled_rows.any():
            mask = (~attention_disabled_rows).view(-1, 1, 1).to(attn_out.dtype)
            attn_out = attn_out * mask
        x = x + attn_out
        ff_input = self._apply_xctx_bias(x, xctx_bias, self.xctx_mlp_gain)
        pre_ff = self.ln2(ff_input)
        pre_ff = self._apply_grce_bias(pre_ff, grce_bias, self.grce_mlp_ld)
        ff_out, mask = self.ff(pre_ff, record_mask=record_mask)
        x = x + ff_out
        return x, cache, mask


def _merge_bias_list(
    bias_list: Sequence[torch.Tensor],
    rows: int,
    cols: int,
    n_layers: int,
    n_embd: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Sum a list of optional bias tensors into a single tensor."""

    if not bias_list:
        return None
    merged = torch.zeros(rows, cols, n_layers, n_embd, device=device, dtype=dtype)
    for bias in bias_list:
        if bias is None:
            continue
        b = bias.to(device=device, dtype=dtype)
        b_rows, b_cols, b_layers, _ = b.shape
        take_cols = min(cols, b_cols)
        take_layers = min(n_layers, b_layers)
        merged[:, :take_cols, :take_layers, :] += b[:, :take_cols, :take_layers, :]
    return merged


def kv_cache_list_merge(
    kv_cache_list: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]] | None],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Concatenate cache segments from multiple sources into per-layer tensors."""

    merged_keys: list[list[torch.Tensor]] = []
    merged_values: list[list[torch.Tensor]] = []
    for source in kv_cache_list:
        if not source:
            continue
        for layer_idx, pair in enumerate(source):
            if pair is None:
                continue
            key, value = pair
            if layer_idx >= len(merged_keys):
                pad = layer_idx + 1 - len(merged_keys)
                merged_keys.extend([[] for _ in range(pad)])
                merged_values.extend([[] for _ in range(pad)])
            merged_keys[layer_idx].append(key)
            merged_values[layer_idx].append(value)
    merged: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    for key_chunks, value_chunks in zip(merged_keys, merged_values):
        if not key_chunks:
            merged.append(None)
            continue
        merged.append((torch.cat(key_chunks, dim=1), torch.cat(value_chunks, dim=1)))
    return merged


def kv_cache_list_detach(
    kv_cache_list: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]] | None],
) -> list[list[tuple[torch.Tensor, torch.Tensor] | None]]:
    """Return a detached copy of every tensor in a kv_cache_list structure."""

    detached: list[list[tuple[torch.Tensor, torch.Tensor] | None]] = []
    for source in kv_cache_list:
        if not source:
            continue
        new_source: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for pair in source:
            if pair is None:
                new_source.append(None)
                continue
            key, value = pair
            new_source.append((key.detach(), value.detach()))
        detached.append(new_source)
    return detached


def kv_cache_list_balance(
    kv_cache_list: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]] | None],
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Collapse many small cache segments into a single chunk per layer."""

    if not kv_cache_list or (kv_cache_list[0] and kv_cache_list[0][0] and
                             len(kv_cache_list) <= kv_cache_list[0][0][0].size(1).bit_length()):
        return kv_cache_list

    total_size = 0
    clog2_buckets = defaultdict(list)
    for i, kv_cache in enumerate(kv_cache_list):
        if not kv_cache or not kv_cache[0]: continue
        n_kv = kv_cache[0][0].size(1)
        if not n_kv: continue
        total_size += n_kv
        clog2_n_kv = n_kv.bit_length()
        clog2_buckets[clog2_n_kv].append((n_kv, i))

    if len(kv_cache_list) == len(clog2_buckets):
        return kv_cache_list

    merge_size = 0
    merge_caches = list()
    final_caches = list()
    for clog2_n_kv, bucket in sorted(clog2_buckets.items()):
        if len(bucket) > 2 or clog2_n_kv == merge_size.bit_length():
            merge_size += sum(n for n, i in bucket)
            merge_caches += bucket
        else:
            final_caches += bucket

    if merge_size:
        merged = kv_cache_list_merge([kv_cache_list[i] for n, i in merge_caches])
        assert merged[0][0].size(1) == merge_size
        final_caches.append((merge_size, None))
    else:
        merged = None

    # return with largest element in position 0 for quick exit on next call
    return [merged if i is None else kv_cache_list[i] for n, i in reversed(sorted(final_caches))]


class RMSNorm(nn.Module):
    """RMS normalization used by the recurrent :class:`TransformerXCTX` path."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        norm = tensor.pow(2).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(norm + self.eps)
        return self.scale * tensor * inv


class TransformerStackCore(nn.Module):
    """Shared Transformer backbone used by both grid and sequence modes."""

    def __init__(self, args: Args) -> None:
        super().__init__()
        config = args
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward_grid(
        self,
        x: torch.Tensor,
        *,
        xctx_bias_list_in: Sequence[torch.Tensor] | None = None,
        grce_bias_list_in: Sequence[torch.Tensor] | None = None,
        kv_cache_list_in: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]] | None = None,
        mode: str = "decode",
        qh_query_callback=None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        rows, cols, _ = x.shape
        xctx_tensor = _merge_bias_list(
            xctx_bias_list_in or [],
            rows,
            cols,
            self.config.n_layer,
            self.config.n_embd,
            x.device,
            x.dtype,
        )
        grce_tensor = _merge_bias_list(
            grce_bias_list_in or [],
            rows,
            cols,
            self.config.n_layer,
            self.config.n_embd,
            x.device,
            x.dtype,
        )
        layer_kv_sources: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(len(self.blocks))
        ]
        if kv_cache_list_in:
            for source in kv_cache_list_in:
                if source is None:
                    continue
                for layer_idx in range(min(len(source), len(self.blocks))):
                    kv_pair = source[layer_idx]
                    if kv_pair is None:
                        continue
                    layer_kv_sources[layer_idx].append(kv_pair)
        current = x
        samples: list[torch.Tensor] = [current]
        kv_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, block in enumerate(self.blocks):
            layer_xctx_bias = None
            if xctx_tensor is not None:
                layer_xctx_bias = xctx_tensor[:, :, layer_idx, :]
            layer_grce_bias = None
            if grce_tensor is not None:
                layer_grce_bias = grce_tensor[:, :, layer_idx, :]
            kv_sources = layer_kv_sources[layer_idx] or None
            current, _, kv_pair = block(
                current,
                xctx_bias=layer_xctx_bias,
                grce_bias=layer_grce_bias,
                kv_cache_sources=kv_sources,
                qh_query_callback=qh_query_callback,
                attn_mode=mode,
                layer_idx=layer_idx,
                full_attention=(mode == "encode"),
            )
            samples.append(current)
            kv_outputs.append(kv_pair if kv_pair is not None else None)
        return current, samples, kv_outputs


class TransformerStackGrid(nn.Module):
    """Evaluate :class:`TransformerStackCore` over short masked grids for GRCE."""

    def __init__(self, core: TransformerStackCore) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        x: torch.Tensor,
        *,
        xctx_bias_list_in: Sequence[torch.Tensor] | None = None,
        grce_bias_list_in: Sequence[torch.Tensor] | None = None,
        kv_cache_list_in: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]] | None = None,
        qh_query_callback=None,
        mode: str = "decode",
    ) -> tuple[torch.Tensor, list[torch.Tensor], list]:
        output, samples, kv_out = self.core.forward_grid(
            x,
            xctx_bias_list_in=xctx_bias_list_in,
            grce_bias_list_in=grce_bias_list_in,
            kv_cache_list_in=kv_cache_list_in,
            mode=mode,
            qh_query_callback=qh_query_callback,
        )
        return output, samples, kv_out


class TransformerGRCE(nn.Module):
    """Implements the GRCE channel invoked by :class:`TransformerStackSequence`."""

    def __init__(self, args: Args) -> None:
        super().__init__()
        config = args
        self.disabled = config.n_grce <= 0
        self.context_dim = config.n_grce
        self.n_layers = config.n_layer
        self.n_embd = config.n_embd
        self.detach_span = max(0, int(config.detach_span))
        if self.disabled:
            return
        self.sample_norms = nn.ModuleList(
            nn.LayerNorm(self.n_embd) for _ in range(self.n_layers)
        )
        self.sample_projections = nn.ModuleList(
            nn.Linear(self.n_embd, self.context_dim) for _ in range(self.n_layers)
        )
        self.bias_norm = nn.LayerNorm(self.context_dim)
        self.bias_projections = nn.ModuleList(
            nn.Linear(self.context_dim, self.n_embd) for _ in range(self.n_layers)
        )
        self.mix_norm = nn.LayerNorm(self.context_dim)
        hidden = max(1, 4 * self.context_dim)
        self.mlp_up = nn.Linear(self.context_dim, hidden)
        self.mlp_down = nn.Linear(hidden, self.context_dim)
        self.output_norm = LayerDampening(self.context_dim, with_gain=False)
        self.dropout = nn.Dropout(config.dropout)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, self.context_dim, device=device, dtype=dtype)

    def bias_forward(self, grce_state: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            raise RuntimeError("GRCE disabled")
        if self.n_layers <= 0:
            return grce_state.new_zeros(grce_state.size(0), 1, 0, self.n_embd)
        normed = self.bias_norm(grce_state)
        per_layer = [proj(normed) for proj in self.bias_projections]
        stacked = torch.stack(per_layer, dim=1)
        return stacked.unsqueeze(1)

    def sample_forward(
        self,
        grce_state: torch.Tensor,
        samples: Sequence[torch.Tensor],
        position: int,
        *,
        detach_samples: bool = False,
    ) -> torch.Tensor:
        if self.disabled:
            return grce_state
        should_detach = detach_samples or (
            self.detach_span > 0 and (position % self.detach_span) == 0
        )
        messages: list[torch.Tensor] = []
        for layer_idx in range(self.n_layers):
            layer_sample = samples[layer_idx][:, -1, :]
            if should_detach:
                layer_sample = layer_sample.detach()
            reduced = self.sample_norms[layer_idx](layer_sample)
            messages.append(self.sample_projections[layer_idx](reduced))
        fused = torch.stack(messages, dim=0).sum(dim=0)
        combined = fused + grce_state
        mixed = self.mix_norm(self.dropout(combined))
        mlp_hidden = F.gelu(self.mlp_up(mixed))
        mlp_out = self.dropout(self.mlp_down(mlp_hidden))
        return self.output_norm(grce_state + fused + mlp_out)

    def parameter_breakdown(self) -> dict[str, int]:
        if self.disabled:
            return {}
        sampler = (
            _module_list_param_count(self.sample_norms)
            + _module_list_param_count(self.sample_projections)
        )
        mlp = (
            _module_param_count(self.mix_norm)
            + _module_param_count(self.mlp_up)
            + _module_param_count(self.mlp_down)
            + _module_param_count(self.output_norm)
        )
        bias = _module_param_count(self.bias_norm) + _module_list_param_count(self.bias_projections)
        return {"samplers": sampler, "mlp": mlp, "bias": bias}


class TransformerXCTX(nn.Module):
    """Implements the XCTX channel consumed by :class:`TransformerStackSequence`."""

    def __init__(self, args: Args) -> None:
        super().__init__()
        config = args
        self.disabled = config.n_xctx <= 0
        self.n_layers = config.n_layer
        self.n_embd = config.n_embd
        self.context_dim = config.n_xctx
        self.inner_dim = _get_inner_xctx_width(config)
        self.squeeze_dim = max(1, self.context_dim // 2)
        self.detach_span = max(0, int(config.detach_span))
        if self.disabled:
            return
        self.sample_linear = nn.ModuleList(
            nn.Linear(self.n_embd, self.inner_dim) for _ in range(self.n_layers)
        )
        self.sample_norms = nn.ModuleList(
            nn.LayerNorm(self.inner_dim) for _ in range(self.n_layers)
        )
        self.expand_linear = nn.ModuleList(
            nn.Linear(self.inner_dim, self.context_dim) for _ in range(self.n_layers)
        )
        self.bias_down = nn.ModuleList(
            nn.Linear(self.context_dim, self.inner_dim) for _ in range(self.n_layers)
        )
        self.bias_norms = nn.ModuleList(
            nn.LayerNorm(self.inner_dim) for _ in range(self.n_layers)
        )
        self.bias_up = nn.ModuleList(
            nn.Linear(self.inner_dim, self.n_embd) for _ in range(self.n_layers)
        )
        self.mix_down = nn.Linear(self.context_dim, self.squeeze_dim)
        self.mix_norm = nn.LayerNorm(self.squeeze_dim)
        self.mix_up = nn.Linear(self.squeeze_dim, self.context_dim)
        self.mix_proj = nn.Linear(self.context_dim, self.context_dim)
        self.output_norm = RMSNorm(self.context_dim)
        self.dropout = nn.Dropout(config.dropout)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, self.context_dim, device=device, dtype=dtype)

    def bias_forward(self, xctx_state: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            raise RuntimeError("XCTX disabled")
        if self.n_layers <= 0:
            return xctx_state.new_zeros(xctx_state.size(0), 1, 0, self.n_embd)
        per_layer: list[torch.Tensor] = []
        for idx in range(self.n_layers):
            reduced = self.bias_down[idx](xctx_state)
            normed = self.bias_norms[idx](reduced)
            per_layer.append(self.bias_up[idx](normed))
        stacked = torch.stack(per_layer, dim=1)
        return stacked.unsqueeze(1)

    def sample_forward(
        self,
        xctx_state: torch.Tensor,
        samples: Sequence[torch.Tensor],
        position: int,
        *,
        detach_samples: bool = False,
    ) -> torch.Tensor:
        if self.disabled:
            return xctx_state
        should_detach = detach_samples or (
            self.detach_span > 0 and (position % self.detach_span) == 0
        )
        messages: list[torch.Tensor] = []
        for idx in range(self.n_layers):
            layer_sample = samples[idx][:, -1, :]
            if should_detach:
                layer_sample = layer_sample.detach()
            centered = layer_sample - layer_sample.mean(dim=-1, keepdim=True)
            reduced = self.sample_linear[idx](centered)
            normed = self.sample_norms[idx](reduced)
            messages.append(self.expand_linear[idx](normed))
        fused = torch.stack(messages, dim=0).sum(dim=0)
        combined = self.dropout(xctx_state + fused)
        squeezed = self.mix_norm(self.mix_down(combined))
        mlp = F.gelu(self.mix_up(squeezed))
        projected = self.dropout(self.mix_proj(mlp))
        mean = projected.mean(dim=-1, keepdim=True)
        updated = xctx_state + (projected - mean)
        return self.output_norm(updated)

    def parameter_breakdown(self) -> dict[str, int]:
        if self.disabled:
            return {}
        sampler = (
            _module_list_param_count(self.sample_linear)
            + _module_list_param_count(self.sample_norms)
            + _module_list_param_count(self.expand_linear)
        )
        bias = (
            _module_list_param_count(self.bias_down)
            + _module_list_param_count(self.bias_norms)
            + _module_list_param_count(self.bias_up)
        )
        mixer = (
            _module_param_count(self.mix_down)
            + _module_param_count(self.mix_norm)
            + _module_param_count(self.mix_up)
            + _module_param_count(self.mix_proj)
            + _module_param_count(self.output_norm)
        )
        return {"samplers": sampler, "bias": bias, "mlp": mixer}


class TransformerStackSequence(nn.Module):
    """Compose the core stack with GRCE/XCTX channels for sequential grids."""

    def __init__(self, args: Args, core: TransformerStackCore) -> None:
        super().__init__()
        self.core = core
        config = args
        self.n_layers = config.n_layer
        self.n_embd = config.n_embd
        self.grce = TransformerGRCE(config) if config.n_grce > 0 else None
        self.xctx = TransformerXCTX(config) if config.n_xctx > 0 else None
        self.kv_rebalance = args.kv_rebalance
        modules = [m for m in (self.grce, self.xctx) if m is not None]
        self.context_modules = nn.ModuleList(modules)

    def _allocate_detached_kv_storage(
        self,
        rows: int,
        cols: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        n_heads = self.core.config.n_head
        head_dim = self.n_embd // n_heads
        storage: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(self.n_layers):
            key_buf = torch.empty(rows, cols, n_heads, head_dim, device=device, dtype=dtype)
            value_buf = torch.empty_like(key_buf)
            storage.append((key_buf, value_buf))
        return storage

    def _detached_kv_prefix(
        self,
        storage: list[tuple[torch.Tensor, torch.Tensor]],
        upto_col: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        if upto_col <= 0:
            return [None] * len(storage)
        prefix: list[tuple[torch.Tensor, torch.Tensor]] = []
        for key_buf, value_buf in storage:
            prefix.append((key_buf[:, :upto_col, :, :], value_buf[:, :upto_col, :, :]))
        return prefix

    def _ensure_state(
        self,
        module: nn.Module | None,
        state: torch.Tensor | None,
        rows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if module is None:
            return None
        if state is None:
            return module.initial_state(rows, device, dtype)
        return state

    def forward(
        self,
        x: torch.Tensor,
        grce_in: torch.Tensor | None = None,
        xctx_in: torch.Tensor | None = None,
        grce_bias_list_in: Sequence[torch.Tensor] | None = None,
        xctx_bias_list_in: Sequence[torch.Tensor] | None = None,
        kv_cache_list_in: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]] | None = None,
        *,
        qh_query_callback=None,
        mode: str = "forward",
        detach_internal_kv_cache: bool = False,
        detach_samples_span: int = 0,
        detach_grce_span: int = 0,
        detach_xctx_span: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list]:
        if mode not in {"forward", "encode", "decode", "noattn"}:
            raise ValueError(f"Unknown TransformerStackSequence mode: {mode}")
        rows, cols, _ = x.shape
        device = x.device
        dtype = x.dtype
        outputs: list[torch.Tensor] = []
        grce_state = self._ensure_state(self.grce, grce_in, rows, device, dtype)
        xctx_state = self._ensure_state(self.xctx, xctx_in, rows, device, dtype)
        if mode in {"encode", "decode"}:
            return self._forward_grid_mode(
                x,
                grce_state,
                xctx_state,
                grce_bias_list_in=grce_bias_list_in,
                xctx_bias_list_in=xctx_bias_list_in,
                kv_cache_list_in=kv_cache_list_in,
                qh_query_callback=qh_query_callback,
                mode=mode,
                detach_samples_span=detach_samples_span,
                detach_grce_span=detach_grce_span,
                detach_xctx_span=detach_xctx_span,
                detach_internal_kv_cache=detach_internal_kv_cache,
            )
        use_internal_cache = mode != "noattn"
        base_sources = list(kv_cache_list_in or [])
        kv_history: list[list[tuple[torch.Tensor, torch.Tensor] | None]] = []
        kv_storage: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if detach_internal_kv_cache:
            kv_storage = self._allocate_detached_kv_storage(rows, cols, device, dtype)
        for col in range(cols):
            column_grce_biases: list[torch.Tensor] = []
            column_xctx_biases: list[torch.Tensor] = []
            if grce_bias_list_in:
                for bias in grce_bias_list_in:
                    if bias is None or bias.size(1) == 0:
                        continue
                    if bias.size(1) == 1:
                        column_grce_biases.append(bias)
                    elif col < bias.size(1):
                        column_grce_biases.append(bias[:, col : col + 1, :, :])
            if xctx_bias_list_in:
                for bias in xctx_bias_list_in:
                    if bias is None or bias.size(1) == 0:
                        continue
                    if bias.size(1) == 1:
                        column_xctx_biases.append(bias)
                    elif col < bias.size(1):
                        column_xctx_biases.append(bias[:, col : col + 1, :, :])
            if self.grce is not None and grce_state is not None:
                column_grce_biases.append(self.grce.bias_forward(grce_state))
            if self.xctx is not None and xctx_state is not None:
                column_xctx_biases.append(self.xctx.bias_forward(xctx_state))
            column_input = x[:, col : col + 1, :]
            column_kv_sources: list[Sequence[tuple[torch.Tensor, torch.Tensor] | None]] = []
            if base_sources:
                column_kv_sources.extend(base_sources)
            if detach_internal_kv_cache and kv_storage is not None and col > 0:
                column_kv_sources.append(self._detached_kv_prefix(kv_storage, col))
            elif (not detach_internal_kv_cache) and use_internal_cache and kv_history:
                column_kv_sources.extend(kv_history)
            kv_sources_arg = column_kv_sources if column_kv_sources else None
            column_output, samples, kv_pairs = self.core.forward_grid(
                column_input,
                xctx_bias_list_in=column_xctx_biases,
                grce_bias_list_in=column_grce_biases,
                kv_cache_list_in=kv_sources_arg,
                mode="decode",
                qh_query_callback=qh_query_callback,
            )
            if detach_internal_kv_cache:
                column_output = column_output.detach()
                samples = [sample.detach() for sample in samples]
            outputs.append(column_output)
            if detach_internal_kv_cache and kv_storage is not None:
                for layer_idx, kv_pair in enumerate(kv_pairs):
                    if kv_pair is None:
                        continue
                    key_chunk, value_chunk = kv_pair
                    key_buf, value_buf = kv_storage[layer_idx]
                    key_buf[:, col : col + key_chunk.size(1), :, :].copy_(key_chunk.detach())
                    value_buf[:, col : col + value_chunk.size(1), :, :].copy_(value_chunk.detach())
            else:
                kv_history.append(kv_pairs)
                if self.kv_rebalance:
                    kv_history = kv_cache_list_balance(kv_history)
            detach_samples = detach_samples_span > 0 and (col % detach_samples_span) == 0
            if self.grce is not None and grce_state is not None:
                if detach_grce_span > 0 and (col % detach_grce_span) == 0:
                    grce_state = grce_state.detach()
                grce_state = self.grce.sample_forward(
                    grce_state,
                    samples,
                    col,
                    detach_samples=detach_samples,
                )
            if self.xctx is not None and xctx_state is not None:
                if detach_xctx_span > 0 and (col % detach_xctx_span) == 0:
                    xctx_state = xctx_state.detach()
                xctx_state = self.xctx.sample_forward(
                    xctx_state,
                    samples,
                    col,
                    detach_samples=detach_samples,
                )
        stacked = torch.cat(outputs, dim=1)
        if detach_internal_kv_cache and kv_storage is not None:
            kv_out_base = [
                (key_buf.detach(), value_buf.detach()) for key_buf, value_buf in kv_storage
            ]
            kv_out = kv_out_base
        else:
            kv_out = kv_cache_list_merge(kv_history)
        return stacked, grce_state, xctx_state, kv_out

    def _forward_grid_mode(
        self,
        x: torch.Tensor,
        grce_state: torch.Tensor | None,
        xctx_state: torch.Tensor | None,
        *,
        grce_bias_list_in: Sequence[torch.Tensor] | None,
        xctx_bias_list_in: Sequence[torch.Tensor] | None,
        kv_cache_list_in: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]] | None,
        qh_query_callback=None,
        mode: str,
        detach_samples_span: int,
        detach_grce_span: int,
        detach_xctx_span: int,
        detach_internal_kv_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]]]:
        rows, cols, _ = x.shape
        grid_grce_biases = list(grce_bias_list_in or [])
        grid_xctx_biases = list(xctx_bias_list_in or [])
        if self.grce is not None and grce_state is not None:
            grid_grce_biases.append(self.grce.bias_forward(grce_state))
        if self.xctx is not None and xctx_state is not None:
            grid_xctx_biases.append(self.xctx.bias_forward(xctx_state))
        output, samples, kv_pairs = self.core.forward_grid(
            x,
            xctx_bias_list_in=grid_xctx_biases,
            grce_bias_list_in=grid_grce_biases,
            kv_cache_list_in=kv_cache_list_in,
            mode=mode,
            qh_query_callback=qh_query_callback,
        )
        kv_out = kv_pairs
        if detach_internal_kv_cache:
            kv_out = [
                None if pair is None else (pair[0].detach(), pair[1].detach())
                for pair in kv_out
            ]
        detach_samples = False
        for col in range(cols):
            if detach_samples_span > 0:
                detach_samples = (col % detach_samples_span) == 0
            slice_samples = [sample[:, : col + 1, :] for sample in samples[: self.n_layers]]
            if self.grce is not None and grce_state is not None:
                if detach_grce_span > 0 and (col % detach_grce_span) == 0:
                    grce_state = grce_state.detach()
                grce_state = self.grce.sample_forward(
                    grce_state,
                    slice_samples,
                    col,
                    detach_samples=detach_samples,
                )
            if self.xctx is not None and xctx_state is not None:
                if detach_xctx_span > 0 and (col % detach_xctx_span) == 0:
                    xctx_state = xctx_state.detach()
                xctx_state = self.xctx.sample_forward(
                    xctx_state,
                    slice_samples,
                    col,
                    detach_samples=detach_samples,
                )
        return output, grce_state, xctx_state, kv_out


@dataclass
class LayerCache:
    """Per-layer KV cache container used by :class:`GPTCore`."""

    segments: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    length: int = 0
    allow_rebalance: bool = True
    use_buffer: bool = False
    key_buffer: torch.Tensor | None = None
    value_buffer: torch.Tensor | None = None
    buffer_position: int = 0

    def configure_detached_buffer(
        self,
        key_buffer: torch.Tensor,
        value_buffer: torch.Tensor,
    ) -> None:
        if key_buffer.shape != value_buffer.shape:
            raise ValueError("Key/value buffers must share the same shape")
        self.segments.clear()
        self.length = 0
        self.buffer_position = 0
        self.use_buffer = True
        self.allow_rebalance = False
        self.key_buffer = key_buffer
        self.value_buffer = value_buffer

    def append(self, key_chunk: torch.Tensor, value_chunk: torch.Tensor) -> None:
        if key_chunk.size(2) != value_chunk.size(2):
            raise ValueError("Key/value chunks must share the same length")
        if self.use_buffer:
            self._append_to_buffer(key_chunk, value_chunk)
            return
        self.segments.append((key_chunk, value_chunk))
        self.length += key_chunk.size(2)
        if self.allow_rebalance:
            self._rebalance_segments()

    def _rebalance_segments(self) -> None:
        if self.use_buffer:
            return
        while len(self.segments) >= 2:
            key_b, value_b = self.segments[-1]
            key_a, value_a = self.segments[-2]
            if key_a.size(2) != key_b.size(2):
                break
            merged_key = torch.cat([key_a, key_b], dim=2)
            merged_value = torch.cat([value_a, value_b], dim=2)
            self.segments.pop()
            self.segments.pop()
            self.segments.append((merged_key, merged_value))

    def _append_to_buffer(self, key_chunk: torch.Tensor, value_chunk: torch.Tensor) -> None:
        if self.key_buffer is None or self.value_buffer is None:
            raise RuntimeError("Detached KV cache buffers not initialized")
        chunk = key_chunk.size(2)
        start = self.buffer_position
        end = start + chunk
        if end > self.key_buffer.size(2):
            raise RuntimeError("Detached KV cache exhausted; increase max sequence length")
        with torch.no_grad():
            self.key_buffer[:, :, start:end, :].copy_(key_chunk.detach())
            self.value_buffer[:, :, start:end, :].copy_(value_chunk.detach())
        self.buffer_position = end
        self.length = end

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_buffer:
            if self.key_buffer is None or self.value_buffer is None or self.length <= 0:
                raise RuntimeError("LayerCache is empty; call append() before tensors().")
            return (
                self.key_buffer[:, :, : self.length, :],
                self.value_buffer[:, :, : self.length, :],
            )
        if not self.segments:
            raise RuntimeError("LayerCache is empty; call append() before tensors().")
        if len(self.segments) == 1:
            return self.segments[0]
        keys = torch.cat([seg[0] for seg in self.segments], dim=2)
        values = torch.cat([seg[1] for seg in self.segments], dim=2)
        return keys, values


class GRCEGPT(nn.Module):
    """Top-level model instantiated by training/eval helpers and :func:`grce_main`."""

    def __init__(self, args: Args) -> None:
        super().__init__()
        self.config = args
        self.core = TransformerStackCore(args)
        self.stack_grid = TransformerStackGrid(self.core)
        self.stack_sequence = TransformerStackSequence(args, self.core)
        self.context_channels = self.stack_sequence.context_modules

    def _position_ids(
        self,
        length: int,
        batch_size: int,
        device: torch.device,
        position_offsets: torch.Tensor | None,
    ) -> torch.Tensor:
        base = torch.arange(length, device=device).unsqueeze(0).expand(batch_size, -1)
        if position_offsets is None:
            return base
        offsets = position_offsets.to(device=device, dtype=torch.long)
        if offsets.dim() != 1 or offsets.shape[0] != batch_size:
            raise ValueError("position_offsets must be 1D with batch_size entries")
        return base + offsets.view(batch_size, 1)

    def forward_autoreg(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        mode: str = "forward",
        position_offsets: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        del targets  # unused but kept for API compatibility
        B, T = idx.shape
        device = idx.device
        pos_idx = self._position_ids(T, B, device, position_offsets)
        if torch.any(pos_idx >= self.config.block_size):
            raise ValueError("position ids exceed configured --block-size")
        tok = self.core.tok_emb(idx)
        pos = self.core.pos_emb(pos_idx)
        x = self.core.drop(tok + pos)
        context_info: dict[str, torch.Tensor] | None = None
        if mode in {"forward", "noattn", "encode"}:
            sequence_output, grce_out, xctx_out, _ = self.stack_sequence.forward(
                x,
                mode=mode,
            )
            hidden = sequence_output
            context_info = {}
            if grce_out is not None:
                context_info["grce"] = grce_out
            if xctx_out is not None:
                context_info["xctx"] = xctx_out
            if not context_info:
                context_info = None
        elif mode == "decode":
            decode_output, _, _ = self.stack_grid.forward(x, mode="decode")
            hidden = decode_output
        else:
            raise ValueError(f"Unknown forward_autoreg mode: {mode}")
        logits = self.core.head(self.core.ln_f(hidden))
        return logits, None, context_info


def build_model_tag(config: GeometryLike) -> str:
    """Build the filename tag used by ``train``/``create`` checkpoints."""

    tag = (
        f"v{config.vocab_size}_bs{config.block_size}_emb{config.n_embd}_"
        f"layers{config.n_layer}_heads{config.n_head}"
    )
    if config.n_grce > 0:
        tag += f"_grce{config.n_grce}"
    if config.n_xctx > 0:
        tag += f"_xctx{config.n_xctx}"
    return tag


LOSS_IGNORE_INDEX = -100


# -----------------------------------------------------------------------------
# Training / Generation Helpers
# -----------------------------------------------------------------------------


BATCH_MODES: tuple[str, ...] = ("encode", "decode", "forward", "noattn")

ROW_METRIC_HIST_KEYS = list(BATCH_MODES)
ROW_METRIC_LOG_KEYS = list(BATCH_MODES)
ROW_METRIC_LOG_GROUP = {"encode", "forward"}


@dataclass
class EvalBatchStats:
    """Holds averaged losses plus raw sums/counts for evaluation batches."""

    metrics: dict[str, float | None]
    loss_sums: dict[str, float]
    token_counts: dict[str, int]


def _combine_eval_stats(stats_list: Sequence[EvalBatchStats]) -> EvalBatchStats:
    """Merge multiple evaluation runs by summing loss totals and counts."""

    keys = list(BATCH_MODES) + ["target"]
    combined_sums = {key: 0.0 for key in keys}
    combined_counts = {key: 0 for key in keys}
    for stats in stats_list:
        for key in keys:
            combined_sums[key] += stats.loss_sums.get(key, 0.0)
            combined_counts[key] += stats.token_counts.get(key, 0)
    metrics = {key: None for key in keys}
    for key in keys:
        count = combined_counts[key]
        if count > 0:
            metrics[key] = combined_sums[key] / count
    return EvalBatchStats(metrics, combined_sums, combined_counts)


@dataclass
class LayoutPassResult:
    total_loss_sum: torch.Tensor | None
    total_tokens: int
    mode_loss_sums: dict[str, float]
    mode_token_counts: dict[str, int]


def _log_layout_warnings(args: Args, layout: BatchLayout) -> None:
    if not layout.warnings:
        return
    cache: set[str] = getattr(args, "_layout_warning_cache", set())
    printed = False
    for warning in layout.warnings:
        if warning in cache:
            continue
        cache.add(warning)
        printed = True
        print(color_text(f"Layout warning: {warning}", Colors.YELLOW))
    if printed:
        args._layout_warning_cache = cache


def _run_microbatch_pass(
    args: Args,
    model: GRCEGPT,
    rows: Sequence[RowLayout],
    micro_window: TokenWindow,
    device: torch.device,
    *,
    collect_mode_metrics: bool,
) -> tuple[LayoutPassResult, float]:
    start_time = time.time()
    mode_loss_sums = {mode: 0.0 for mode in BATCH_MODES}
    mode_token_counts = {mode: 0 for mode in BATCH_MODES}
    total_loss_sum: torch.Tensor | None = None
    total_tokens = 0
    for group in rows:
        rows = int(group.rows)
        if rows <= 0:
            continue
        cols_total = group.total_columns()
        if cols_total <= 0:
            continue
        try:
            xb, yb = micro_window.sample_batch(cols_total, rows, device)
        except ValueError as exc:
            raise ValueError(
                f"Unable to sample {rows} rows with {cols_total} columns from the current window"
            ) from exc
        embeddings = _sequence_embeddings(model, xb)
        cursor = 0
        kv_chain: list[list[tuple[torch.Tensor, torch.Tensor]] | None] = []
        grce_state = None
        xctx_state = None
        for segment in group.segments:
            cols = int(segment.columns)
            if cols <= 0:
                continue
            mode = segment.mode
            chunk_input = embeddings[:, cursor : cursor + cols, :]
            chunk_target = yb[:, cursor : cursor + cols]
            kv_sources = None if mode == "noattn" else (kv_chain if kv_chain else None)
            chunk_output, grce_state, xctx_state, kv_out = model.stack_sequence.forward(
                chunk_input,
                grce_in=grce_state,
                xctx_in=xctx_state,
                kv_cache_list_in=kv_sources,
                mode=mode,
                detach_internal_kv_cache=args.detach_kv_cache,
            )
            logits = model.core.head(model.core.ln_f(chunk_output))
            loss_sum, token_count = loss_sum_and_token_count(
                logits,
                chunk_target,
                last_only=(mode == "encode"),
            )
            if token_count > 0:
                total_tokens += token_count
                total_loss_sum = loss_sum if total_loss_sum is None else total_loss_sum + loss_sum
                if collect_mode_metrics:
                    mode_loss_sums[mode] += float(loss_sum.detach().item())
                    mode_token_counts[mode] += token_count
            if mode != "noattn":
                kv_chain.append(kv_out)
            cursor += cols
    result = LayoutPassResult(total_loss_sum, total_tokens, mode_loss_sums, mode_token_counts)
    return result, time.time() - start_time


def train_layout_batch(
    args: Args,
    model: GRCEGPT,
    dataset: TextDataset,
    layout: BatchLayout,
    device: torch.device,
    grad_hook: Callable[[int, int], None] | None = None,
) -> tuple[torch.Tensor, int, list[tuple[int, float, float, str]], dict[str, object]]:
    step_span = layout.total_token_span()
    if step_span <= 0:
        raise ValueError("Layout produced zero tokens for training step")
    window_rng = random.Random()
    step_window = dataset.sample_window("train", step_span, rng=window_rng)
    total_loss_sum: torch.Tensor | None = None
    total_tokens = 0
    micro_logs: list[tuple[int, float, float, str]] = []
    detail_entries: list[dict[str, int]] = []
    head_offset = step_window.start
    for index, batch in enumerate(layout.micro_batches, start=1):
        micro_span = sum(row.token_span() for row in batch)
        if micro_span <= 0:
            micro_logs.append((index, 0.0, 0.0, layout.serialize_rows(batch)))
            continue
        micro_window = step_window.subwindow(micro_span)
        detail_entries.append(
            {
                "micro_index": index,
                "token_start": micro_window.start,
                "token_end": micro_window.start + micro_window.length,
            }
        )
        result, fwd_time = _run_microbatch_pass(
            args,
            model,
            batch,
            micro_window,
            device,
            collect_mode_metrics=False,
        )
        if result.total_loss_sum is None or result.total_tokens <= 0:
            micro_logs.append((index, fwd_time, 0.0, layout.serialize_rows(batch)))
            continue
        bwd_start = time.time()
        result.total_loss_sum.backward()
        if grad_hook is not None:
            grad_hook(index, micro_span)
        bwd_time = time.time() - bwd_start
        micro_logs.append((index, fwd_time, bwd_time, layout.serialize_rows(batch)))
        total_loss_sum = (
            result.total_loss_sum
            if total_loss_sum is None
            else total_loss_sum + result.total_loss_sum
        )
        total_tokens += result.total_tokens
    if total_loss_sum is None:
        raise RuntimeError("Layout batch produced no tokens")
    meta_entry: dict[str, object] = {
        "step_start": head_offset,
        "step_end": head_offset + step_window.length,
        "micro": detail_entries,
    }
    return total_loss_sum, total_tokens, micro_logs, meta_entry


def evaluate_layout_batch(
    args: Args,
    model: GRCEGPT,
    dataset: TextDataset,
    split: str,
    layout: BatchLayout,
    device: torch.device,
) -> EvalBatchStats:
    step_span = layout.total_token_span()
    aggregate_loss_sums = {mode: 0.0 for mode in BATCH_MODES}
    aggregate_token_counts = {mode: 0 for mode in BATCH_MODES}
    total_loss_sum: torch.Tensor | None = None
    total_tokens = 0
    if step_span <= 0:
        metrics: dict[str, float | None] = {mode: None for mode in BATCH_MODES}
        metrics["target"] = None
        loss_sums = {mode: 0.0 for mode in BATCH_MODES}
        loss_sums["target"] = 0.0
        token_counts = {mode: 0 for mode in BATCH_MODES}
        token_counts["target"] = 0
        return EvalBatchStats(metrics, loss_sums, token_counts)
    window_rng = random.Random()
    step_window = dataset.sample_window(split, step_span, rng=window_rng)
    for batch_rows in layout.micro_batches:
        micro_span = sum(row.token_span() for row in batch_rows)
        if micro_span <= 0:
            continue
        micro_window = step_window.subwindow(micro_span)
        result, _ = _run_microbatch_pass(
            args,
            model,
            batch_rows,
            micro_window,
            device,
            collect_mode_metrics=True,
        )
        if result.total_loss_sum is not None:
            total_loss_sum = (
                result.total_loss_sum
                if total_loss_sum is None
                else total_loss_sum + result.total_loss_sum
            )
        total_tokens += result.total_tokens
        for mode in BATCH_MODES:
            aggregate_loss_sums[mode] += result.mode_loss_sums.get(mode, 0.0)
            aggregate_token_counts[mode] += result.mode_token_counts.get(mode, 0)
    metrics: dict[str, float | None] = {mode: None for mode in BATCH_MODES}
    loss_sums = {mode: aggregate_loss_sums[mode] for mode in BATCH_MODES}
    token_counts = {mode: aggregate_token_counts[mode] for mode in BATCH_MODES}
    total_loss_value = float(total_loss_sum.item()) if total_loss_sum is not None else 0.0
    for mode in BATCH_MODES:
        count = token_counts[mode]
        if count > 0:
            metrics[mode] = loss_sums[mode] / count
    metrics["target"] = None if total_tokens <= 0 else total_loss_value / total_tokens
    loss_sums["target"] = total_loss_value
    token_counts["target"] = total_tokens
    return EvalBatchStats(metrics, loss_sums, token_counts)


def count_eval_calls(steps: int, interval: int) -> int:
    """Return how many evaluation batches run in a training cycle."""

    if steps <= 0:
        return 0
    eval_steps = {1, steps}
    if interval > 0:
        current = interval
        while current <= steps:
            eval_steps.add(current)
            current += interval
    return len(eval_steps)


def evaluation_step_indices(steps: int, interval: int) -> list[int]:
    """List the training steps that trigger evaluation passes."""

    if steps <= 0:
        return []
    eval_steps: set[int] = {1, steps}
    if interval > 0:
        current = interval
        while current <= steps:
            eval_steps.add(current)
            current += interval
    return sorted(eval_steps)


def loss_sum_and_token_count(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    last_only: bool = False,
) -> tuple[torch.Tensor, int]:
    if last_only:
        logits = logits[:, -1:, :]
        targets = targets[:, -1:]
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        ignore_index=LOSS_IGNORE_INDEX,
    ).view(targets.size(0), -1)
    valid_mask = (targets != LOSS_IGNORE_INDEX).to(per_token.dtype)
    loss_sum = (per_token * valid_mask).sum()
    token_count = int(valid_mask.sum().item())
    return loss_sum, token_count


def _sequence_embeddings(model: GRCEGPT, token_batch: torch.Tensor) -> torch.Tensor:
    """Project token IDs into dropout'd embeddings for stack sequence calls."""

    batch_size, seq_len = token_batch.shape
    pos_idx = model._position_ids(seq_len, batch_size, token_batch.device, None)
    tok = model.core.tok_emb(token_batch)
    pos = model.core.pos_emb(pos_idx)
    return model.core.drop(tok + pos)


def _scale_gradients(module: nn.Module, scale: float) -> None:
    if scale == 1.0:
        return
    for param in module.parameters():
        if param.grad is not None:
            param.grad.mul_(scale)


def _grad_norm(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += grad.pow(2).sum().item()
    return math.sqrt(total)




def train_model(
    args: Args,
    model: GRCEGPT,
    dataset: TextDataset,
    device: torch.device,
    steps: int,
    block_length: int,
    batch_size: int,
    eval_interval: int,
    start_step: int,
    optimizer: torch.optim.Optimizer,
    sample_prompt: torch.Tensor,
    sample_chars: int,
    tokenizer: GPT2TokenizerWrapper,
    suppress_newlines: bool,
    newline_token_id: int | None,
    prompt_registry: PromptRegistry | None = None,
    *,
    cycle_wall_start: float,
    base_wall_seconds: float,
    show_time: bool = False,
    default_prompt_boundary: bool = False,
    boundary_blocklist: Sequence[int] | None = None,
    show_train_loss_details: bool = False,
    show_test_loss_details: bool = True,
    prebuilt_layouts: Sequence[BatchLayout] | None = None,
) -> Tuple[int, List[Dict[str, float]], float, float]:
    """Run the main training loop for a cycle."""

    if optimizer is None:
        raise ValueError("train_model requires an initialized optimizer instance")
    total_steps = start_step
    history_updates: List[Dict[str, float]] = []
    printed_header = False
    eval_interval = max(1, int(eval_interval))
    loop_timer = Timer().start()
    eval_timer = Timer()

    long_loss_header = " ".join([""] + [f"{': ' if key in ROW_METRIC_LOG_GROUP else ''}{key}" for key in ROW_METRIC_LOG_KEYS])

    line_parts: List[str] = []
    if show_time:
        line_parts.append(color_text("time", Colors.BLUE))
    line_parts.append(color_text(f"step", Colors.CYAN))
    line_parts.append(color_text("train" + (long_loss_header if show_train_loss_details else ""), Colors.MAGENTA))
    line_parts.append(color_text("test" + (long_loss_header if show_test_loss_details else ""), Colors.GREEN))
    line = " | ".join(line_parts) + " |"
    print(line)

    oom_retries = 0
    step = 0
    layouts_sequence = list(prebuilt_layouts) if prebuilt_layouts is not None else None
    manual_layout_override: BatchLayout | None = None
    micro_grad_norms: list[tuple[int, int, float]] = []
    cycle_micro_norms: list[float] = []
    cycle_step_norms: list[float] = []
    need_grad_tracking = args.log_grad_norms or args.grad_summary
    while step < steps:
        if layouts_sequence is not None:
            if step >= len(layouts_sequence):
                raise ValueError("Not enough precomputed layouts for this cycle")
            layout = layouts_sequence[step]
        elif manual_layout_override is not None:
            layout = manual_layout_override
            manual_layout_override = None
        else:
            layout = BatchLayout(args.layout, batch_size=batch_size, block_size=block_length)
        layout_serialized = layout.serialize()
        layout_span = layout.total_token_span()
        _log_layout_warnings(args, layout)
        current_step_index = total_steps + 1
        step_wall_start = time.time()
        try:
            optimizer.zero_grad(set_to_none=True)
            micro_grad_norms.clear()
            def _record_micro_grad(micro_idx: int, micro_tokens: int) -> None:
                raw_norm = _grad_norm(model)
                norm = raw_norm / max(1, micro_tokens)
                if args.log_grad_norms:
                    micro_grad_norms.append((micro_idx, micro_tokens, raw_norm, norm))
                if args.grad_summary:
                    cycle_micro_norms.append(norm)
            total_loss_sum, total_tokens, micro_logs, window_detail = train_layout_batch(
                args,
                model,
                dataset,
                layout,
                device,
                grad_hook=_record_micro_grad if need_grad_tracking else None,
            )
            if total_tokens <= 0:
                raise RuntimeError("No tokens processed in training step")
            opt_start = time.time()
            grad_scale = 1.0 / float(total_tokens)
            _scale_gradients(model, grad_scale)
            step_grad_norm = _grad_norm(model) if need_grad_tracking else None
            if args.grad_summary and step_grad_norm is not None:
                cycle_step_norms.append(step_grad_norm)
            optimizer.step()
            opt_duration = time.time() - opt_start
            total_loss = total_loss_sum / float(total_tokens)
        except torch.OutOfMemoryError:
            oom_retries += 1
            line_parts: List[str] = []
            if show_time:
                timestamp = time.strftime("%H:%M", time.localtime())
                line_parts.append(color_text(timestamp, Colors.BLUE))
            line_parts.append(color_text(f"{total_steps}", Colors.CYAN))
            line_parts.append(color_text(
                f"OOM (retry {oom_retries}/3) during layout {layout_serialized}; "
                "refreshing layout and retrying",
                Colors.YELLOW,
            ))
            line = " | ".join(line_parts)
            print(line)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            if oom_retries >= 3:
                raise
            max_span = layout_span
            replacement = layout
            attempts = 0
            while True:
                candidate = BatchLayout(args.layout, batch_size=batch_size, block_size=block_length)
                candidate_span = candidate.total_token_span()
                replacement = candidate
                if max_span <= 0 or candidate_span <= max_span or attempts >= 8:
                    break
                attempts += 1
            if layouts_sequence is not None:
                layouts_sequence[step] = replacement
            else:
                manual_layout_override = replacement
            continue
        step_wall = time.time() - step_wall_start
        if args.log_grad_norms and micro_grad_norms:
            norms = [value for _, _, _, value in micro_grad_norms]
            stats_line = (
                f"micro grad norms: min {min(norms):.4f}, max {max(norms):.4f},"
                f" avg {sum(norms)/len(norms):.4f}"
            )
            detail_lines = [
                f"  micro {idx}: {raw:.4f} grad / {tokens} tokens = {value:.4f}"
                for idx, tokens, raw, value in micro_grad_norms
            ]
            step_norm_desc = (
                f" step grad norm {step_grad_norm:.4f}" if step_grad_norm is not None else ""
            )
            print(color_text(stats_line + step_norm_desc, Colors.MAGENTA))
            for entry in detail_lines:
                print(color_text(entry, Colors.MAGENTA))
            cycle_micro_norms.extend(norms)
            if step_grad_norm is not None:
                cycle_step_norms.append(step_grad_norm)
        if args.log_step_details:
            step_window_desc = (
                f"tokens {window_detail['step_start']} - {window_detail['step_end']}"
                if isinstance(window_detail, dict)
                else ""
            )
            header = f"Step {current_step_index}: u-batch | fwd | bwd | "
            header += f"(" + step_window_desc + ")  " if step_window_desc else ""
            print(color_text(header + "layout", Colors.BLUE))
            total_fwd = 0.0
            total_bwd = 0.0
            micro_meta = window_detail.get("micro") if isinstance(window_detail, dict) else None
            for idx, fwd_time, bwd_time, rows_str in micro_logs:
                total_fwd += fwd_time
                total_bwd += bwd_time
                token_desc = ""
                if isinstance(micro_meta, list) and 0 < idx <= len(micro_meta):
                    entry = micro_meta[idx - 1]
                    token_desc = f"(offset {entry['token_start']:5})  "
                line = (
                    f"  {idx} | {fwd_time:.2f}s | {bwd_time:.2f}s |"
                    f" {token_desc}{rows_str or layout_serialized}"
                )
                print(color_text(line, Colors.BLUE))
            other_time = max(0.0, step_wall - (total_fwd + total_bwd + opt_duration))
            summary = f"  {opt_duration:.2f}s optimize, {other_time:.2f}s other"
            print(color_text(summary, Colors.BLUE))
        oom_retries = 0
        step += 1
        total_steps += 1

        eval_due = step == 1 or step % eval_interval == 0 or step == steps
        if not eval_due:
            continue
        eval_timer.start()
        model.eval()
        eval_metrics: dict[str, EvalBatchStats] = {}
        with torch.no_grad():
            for split in ("train", "test"):
                eval_metrics[split] = evaluate_layout_batch(
                    args,
                    model,
                    dataset,
                    split,
                    layout,
                    device,
                )
        model.train()
        eval_timer.stop()

        prompt_input = sample_prompt
        selected_prompt_text = args.prompt
        selected_expected_text: str | None = None
        selected_is_default = True
        use_argmax_completion = random.random() < 0.5
        sampling_strategy = "argmax" if use_argmax_completion else "sample"
        if prompt_registry is not None:
            mode = "argmax" if use_argmax_completion else "sample"
            picked_prompt, expected_text, used_default = prompt_registry.pick_prompt(
                mode,
                args.prompt,
            )
            selected_prompt_text = picked_prompt
            selected_expected_text = expected_text
            selected_is_default = used_default
            if used_default:
                prompt_input = sample_prompt
            else:
                prompt_tokens = tokenizer.encode(picked_prompt).unsqueeze(0).to(device)
                prompt_input = prompt_tokens
            prompt_needs_boundary_flag = (
                boundary_blocklist is not None and prompt_needs_boundary(picked_prompt)
            )
        else:
            prompt_needs_boundary_flag = (
                default_prompt_boundary and boundary_blocklist is not None
            )
        sample_tokens, prompt_len = generate(
            model,
            prompt_input.clone(),
            sample_chars,
            suppress_newlines=suppress_newlines,
            newline_token_id=newline_token_id,
            first_token_blocklist=(boundary_blocklist if prompt_needs_boundary_flag else None),
            sampling_strategy=sampling_strategy,
        )
        sample_ids = sample_tokens[0].detach().cpu().tolist()
        prompt_ids = sample_ids[:prompt_len]
        completion_ids = sample_ids[prompt_len:]

        if (
            prompt_registry is not None
            and not selected_is_default
            and selected_expected_text
        ):
            matched = prompt_registry.record_result(
                selected_prompt_text,
                selected_expected_text,
                used_argmax=use_argmax_completion,
                completion_ids=completion_ids,
            )
            if matched:
                mode = "argmax" if use_argmax_completion else "sample"
                print(
                    color_text(
                        f"Prompt satisfied ({mode}): {selected_prompt_text} (expected '{selected_expected_text}')",
                        Colors.YELLOW,
                        bold=True,
                    )
                )

        prompt_text = tokenizer.decode_pretty(args, torch.tensor(prompt_ids))
        completion_text = tokenizer.decode_pretty(args, torch.tensor(completion_ids), alt=True)
        sample_prefix = color_text(prompt_text, Colors.CYAN)
        sample_suffix = color_text(completion_text, Colors.YELLOW)
        sample_render = (Colors.YELLOW if sampling_strategy == 'argmax' else Colors.CYAN) + \
                        f"{sampling_strategy}:{Colors.RESET} " + sample_prefix + sample_suffix

        def format_metric(dataset_split: str, key: str) -> str:
            value = eval_metrics[dataset_split].metrics.get(key)
            if key in ROW_METRIC_LOG_GROUP:
                sep = ": "
            else:
                sep = ""
            if value is None:
                return f"{sep}****"
            return f"{sep}{value:.2f}"

        detail_keys = ROW_METRIC_LOG_KEYS

        def format_train_line() -> str:
            base = format_metric("train", "target")
            if not show_train_loss_details:
                return base
            diag = " ".join(
                format_metric("train", key) for key in detail_keys
            )
            return f"{base} {diag}"

        def format_test_line() -> str:
            base = format_metric("test", "target")
            if not show_test_loss_details:
                return base
            diag = " ".join(
                format_metric("test", key) for key in detail_keys
            )
            return f"{base} {diag}"

        train_values = format_train_line()
        test_values = format_test_line()
        line_parts: List[str] = []
        if show_time:
            timestamp = time.strftime("%H:%M", time.localtime())
            line_parts.append(color_text(timestamp, Colors.BLUE))
        line_parts.append(color_text(f"{total_steps}", Colors.CYAN))
        line_parts.append(color_text(train_values, Colors.MAGENTA))
        line_parts.append(color_text(test_values, Colors.GREEN))
        line = " | ".join(line_parts) + " | " + sample_render
        print(line)

        eval_now = time.time()
        cycle_wall_elapsed = max(0.0, eval_now - cycle_wall_start)
        total_wall_seconds = base_wall_seconds + cycle_wall_elapsed

        record = {
            "step": total_steps,
            "train_loss": float(eval_metrics["train"].metrics.get("target", 0.0) or 0.0),
            "test_loss": float(eval_metrics["test"].metrics.get("target", 0.0) or 0.0),
            "train_wall_seconds": float(total_wall_seconds),
            "unix_time": float(eval_now),
            "train_cursor": int(dataset.positions.get("train", 0)),
            "test_cursor": int(dataset.positions.get("test", 0)),
        }
        record["corpus"] = args.corpus
        record["train_cycle"] = int(dataset.cycles.get("train", 0))
        record["test_cycle"] = int(dataset.cycles.get("test", 0))
        record["batch_layout"] = layout_serialized
        metric_keys = ["target"] + ROW_METRIC_HIST_KEYS
        for key in metric_keys:
            train_val = eval_metrics["train"].metrics.get(key)
            test_val = eval_metrics["test"].metrics.get(key)
            if train_val is not None:
                record[f"train_loss_{key}"] = float(train_val)
            if test_val is not None:
                record[f"test_loss_{key}"] = float(test_val)
        history_updates.append(record)

    if args.grad_summary:
        parts = [color_text(f"[grad norms] ", Colors.CYAN)]
        if cycle_micro_norms:
            parts.append(color_text(
                f"micro: min {min(cycle_micro_norms):.4f}, max {max(cycle_micro_norms):.4f}, "
                f"avg {sum(cycle_micro_norms)/len(cycle_micro_norms):.4f}; ", Colors.MAGENTA))
        if cycle_step_norms:
            parts.append(color_text(
                f"steps: min {min(cycle_step_norms):.4f}, max {max(cycle_step_norms):.4f}, "
                f"avg {sum(cycle_step_norms)/len(cycle_step_norms):.4f}", Colors.GREEN))
        print("".join(parts))
    return total_steps, history_updates, loop_timer.stop(), eval_timer


def run_profile_mode(
    args: Args,
    dataset: TextDataset,
    model: GRCEGPT,
    optimizer: torch.optim.Optimizer,
    *,
    block_length: int,
    batch_size: int,
    device: torch.device,
    block_size: int,
) -> None:
    """Warm up once, profile a second training step, and report CUDA stats."""

    try:
        from torch.profiler import ProfilerActivity, profile, record_function
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "torch.profiler is unavailable; upgrade to PyTorch 1.8+ to use 'profile'."
        ) from exc

    profile_layout = BatchLayout(args.layout, batch_size=batch_size, block_size=block_length)
    _log_layout_warnings(args, profile_layout)
    step_span = profile_layout.total_token_span()
    if step_span <= 0:
        step_span = (block_length + 1) * batch_size
    train_chars = step_span * 2
    dataset.prepare_cycle("train", train_chars)

    def train_step(tag: str, layout: BatchLayout) -> float:
        model.train()
        total_loss_sum, total_tokens, _, _ = train_layout_batch(
            args,
            model,
            dataset,
            layout,
            device,
        )
        if total_tokens <= 0:
            raise RuntimeError("No tokens processed during profiling step")
        loss = total_loss_sum / float(total_tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    warm_loss = train_step("warmup", profile_layout)
    print(color_text(f"Warm-up step loss: {warm_loss:.4f}", Colors.CYAN))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("profile_train_step"):
            prof_loss = train_step("profile", profile_layout)

    print(color_text(f"Profiled step loss: {prof_loss:.4f}", Colors.CYAN))
    cuda_events = sum(
        1 for evt in prof.events() if getattr(evt, "device_type", None) == ProfilerActivity.CUDA
    )
    print(color_text(f"CUDA kernel launches: {cuda_events}", Colors.MAGENTA))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


@torch.no_grad()
def generate(
    model: GRCEGPT,
    idx: torch.Tensor,
    steps: int,
    *,
    suppress_newlines: bool = False,
    newline_token_id: int | None = None,
    first_token_blocklist: Sequence[int] | None = None,
    sampling_strategy: str = "sample",
) -> tuple[torch.Tensor, int]:
    """Autoregressively sample tokens for CLI reports and prompt tests."""

    model.eval()
    idx = idx.clone()
    prompt_len = idx.size(1)
    enforce_first_token_guard = bool(first_token_blocklist)
    blocklist = list(first_token_blocklist or [])
    for _ in range(steps):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _, _ = model.forward_autoreg(idx_cond)
        logits_last = logits[:, -1, :]
        probs = F.softmax(logits_last, dim=-1)
        suppressed_ids: list[int] = []
        if suppress_newlines and newline_token_id is not None:
            suppressed_ids.append(int(newline_token_id))
        if enforce_first_token_guard and blocklist:
            suppressed_ids.extend(blocklist)
        if suppressed_ids:
            modified = probs.clone()
            modified[:, suppressed_ids] = 0
            sums = modified.sum(dim=-1, keepdim=True)
            mask = sums.squeeze(-1) > 0
            if mask.any():
                probs[mask] = modified[mask] / sums[mask]
        if sampling_strategy == "argmax":
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
        if enforce_first_token_guard:
            enforce_first_token_guard = False
    return idx, prompt_len


def run_report_mode(
    model: GRCEGPT,
    tokenizer: GPT2TokenizerWrapper,
    prompt_tokens: torch.Tensor,
    sample_len: int,
    count: int,
    device: torch.device,
    suppress_newlines: bool,
    newline_token_id: int | None,
    default_prompt_boundary: bool,
    boundary_blocklist: Sequence[int] | None,
) -> None:
    """Emit CLI prompt samples used by ``train --report`` and prompt tools."""

    prompt_tokens = prompt_tokens.to(device)
    prompt_text = tokenizer.decode(prompt_tokens[0])
    needs_boundary = default_prompt_boundary and boundary_blocklist is not None
    for idx in range(count):
        generated, prompt_len = generate(
            model,
            prompt_tokens.clone(),
            sample_len,
            suppress_newlines=suppress_newlines,
            newline_token_id=newline_token_id,
            first_token_blocklist=(boundary_blocklist if needs_boundary else None),
            sampling_strategy="sample" if idx % 2 else "argmax",
        )
        sample_ids = generated[0].tolist()
        completion_ids = sample_ids[prompt_len:]
        completion = tokenizer.decode(torch.tensor(completion_ids))
        print(color_text(f"Sample #{idx + 1}: {prompt_text}{completion}", Colors.GREEN))


def run_test_slice(
    args: Args,
    dataset: TextDataset,
    tokenizer: GPT2TokenizerWrapper,
    model: GRCEGPT,
    block_length: int,
    start_pos: int,
) -> None:
    """Print a colored snippet from the test set for ``train --test-slice``."""

    tokens = dataset.looped_slice("test", start_pos, block_length)
    pretty_text = tokenizer.decode_pretty(args, tokens)
    print(color_text(f"Test slice @ {start_pos}:", Colors.CYAN))
    print(pretty_text)


def preprocess_runtime_args(args: Args) -> None:
    """Resolve checkpoint overrides and derived paths before runtime spins up."""

    if getattr(args, "_checkpoint_preprocessed", False):
        return
    args._checkpoint_preprocessed = True

    args.model_path_override = None
    args.log_path_override = None
    args.checkpoint_payload_override = None
    args.tokenizer_json_override = None

    if args.pt:
        checkpoint_path = args.pt
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_config = payload.get("config")
        if saved_config is None:
            raise ValueError(
                "Checkpoint lacks config metadata; re-save it with the latest format."
            )
        saved = dict(saved_config)
        legacy_xctx = bool(saved.pop("grce_xctx", False))
        if "n_xctx" not in saved:
            if legacy_xctx:
                saved["n_xctx"] = int(saved.get("n_grce", 0))
                saved["n_grce"] = 0
            else:
                saved["n_xctx"] = 0
        config = ModelGeometry(**saved)
        args.checkpoint_payload_override = payload
        args.tokenizer_json_override = payload.get("tokenizer_json")
        args.block_size = config.block_size
        if not getattr(args, "_block_length_defined", False):
            args.block_length = config.block_size
        elif args.block_length > config.block_size:
            raise ValueError("--block-length cannot exceed checkpoint block size")
        args.n_layer = config.n_layer
        args.n_head = config.n_head
        args.n_embd = config.n_embd
        args.n_grce = config.n_grce
        args.n_xctx = config.n_xctx
        args.vocab_size = config.vocab_size
        args.model_path_override = checkpoint_path
        args.log_path_override = checkpoint_path.with_suffix(".log")
        return

    inferred = ModelGeometry(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_grce=args.n_grce,
        n_xctx=args.n_xctx,
    )
    tag = build_model_tag(inferred)
    for extra_tag in args.tag:
        cleaned = re.sub(r"[^0-9A-Za-z]+", "", extra_tag)
        if cleaned:
            tag += f"_{cleaned}"
    model_dir = pathlib.Path(args.model)
    prefix = f"{args.name}_model_"
    model_path = model_dir / f"{prefix}{tag}.pt"
    args.model_path_override = model_path
    args.log_path_override = model_path.with_suffix(".log")


import signal
import traceback

class Runtime:
    """Coordinates CLI commands and training loops for :func:`grce_main`."""

    def __init__(self, args: Args):
        self.args = args
        self.tokenizer: GPT2TokenizerWrapper | None = None
        self.dataset: TextDataset | None = None
        self.newline_token_id: int | None = None
        self.boundary_blocklist: Sequence[int] | None = None
        self.default_prompt_boundary: bool = False
        self.model_path: pathlib.Path | None = None
        self.log_path: pathlib.Path | None = None
        self.tokenizer_json: str | None = None
        self.datasets_state: dict[str, dict[str, int]] = {}

    class TimeoutAlarm(Exception):
        pass

    def cancel_timeout(self) -> None:
        pass

    def start_timeout(self, timeout_seconds) -> None:
        self.cancel_timeout()

        timeout_method: str | None = None
        prev_sigalrm_handler = None

        def cancel_timeout() -> None:
            nonlocal timeout_method, prev_sigalrm_handler
            if timeout_method == "setitimer":
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            elif timeout_method == "alarm":
                signal.alarm(0)
            if timeout_method is not None:
                handler = prev_sigalrm_handler or signal.SIG_DFL
                signal.signal(signal.SIGALRM, handler)
            timeout_method = None
            prev_sigalrm_handler = None

        if timeout_seconds > 0:
            def handle_timeout(signum: int, frame: object) -> None:
                raise self.TimeoutAlarm()

            prev_sigalrm_handler = signal.signal(signal.SIGALRM, handle_timeout)
            try:
                signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
                timeout_method = "setitimer"
            except AttributeError:
                timeout_method = "alarm"
                signal.alarm(max(1, int(math.ceil(timeout_seconds))))

        self.cancel_timeout = cancel_timeout

    def _prepare_tokenizer_bundle(
        self,
        *,
        allow_files: bool,
    ) -> tuple[
        GPT2TokenizerWrapper,
        int | None,
        Sequence[int] | None,
        bool,
        str,
    ]:
        tokenizer_json = getattr(self.args, "tokenizer_json_override", None)
        if tokenizer_json is None:
            if not allow_files:
                raise RuntimeError("Tokenizer JSON not available; run 'create' to refresh checkpoint metadata.")
            data_dir = pathlib.Path(self.args.data)
            tokenizer_key = f"vocab_{self.args.vocab_size}"
            tokenizer_path = (
                pathlib.Path(self.args.tokenizer)
                if self.args.tokenizer
                else (data_dir / f"{tokenizer_key}.json")
            )
            if not tokenizer_path.exists():
                raise FileNotFoundError(
                    f"Tokenizer JSON {tokenizer_path} not found; run corpus.py tokenizer to create it."
                )
            tokenizer_json = tokenizer_path.read_text(encoding="utf-8")
            self.args.tokenizer_json_override = tokenizer_json
            print(color_text(f"Tokenizer: {tokenizer_path}", Colors.BLUE))
        tokenizer = GPT2TokenizerWrapper(
            tokenizer_json=tokenizer_json,
        )
        self.args.vocab_size = tokenizer.vocab_size

        newline_token_id = None
        newline_tokens = tokenizer.encode_ids("\n")
        if newline_tokens:
            newline_token_id = newline_tokens[0]

        enforce_boundary_guard = not self.args.no_boundary
        boundary_blocklist = (
            tokenizer.leading_alpha_token_ids if enforce_boundary_guard else None
        )
        default_prompt_boundary = (
            enforce_boundary_guard
            and boundary_blocklist is not None
            and prompt_needs_boundary(self.args.prompt)
        )

        self.tokenizer = tokenizer
        self.newline_token_id = newline_token_id
        self.boundary_blocklist = boundary_blocklist
        self.default_prompt_boundary = default_prompt_boundary
        self.tokenizer_json = tokenizer_json

        return (
            tokenizer,
            newline_token_id,
            boundary_blocklist,
            default_prompt_boundary,
            tokenizer_json,
        )

    def _prepare_corpus(
        self,
    ) -> tuple[
        GPT2TokenizerWrapper,
        TextDataset,
        int | None,
        Sequence[int] | None,
        bool,
        str,
    ]:
        (
            tokenizer,
            newline_token_id,
            boundary_blocklist,
            default_prompt_boundary,
            tokenizer_json,
        ) = self._prepare_tokenizer_bundle(allow_files=False)
        if not self.args.corpus:
            raise RuntimeError("No corpus configured; run 'grce.py corpus --set <name>' first.")
        data_dir = pathlib.Path(self.args.data)
        train_cache_path = data_dir / f"{self.args.corpus}_tokens_train_{self.args.vocab_size}.pt"
        test_cache_path = data_dir / f"{self.args.corpus}_tokens_test_{self.args.vocab_size}.pt"
        train_tokens = load_cached_tokens("train", train_cache_path)
        test_tokens = load_cached_tokens("test", test_cache_path)

        train_token_count = int(train_tokens.numel())
        test_token_count = int(test_tokens.numel())
        print(
            color_text(
                (
                    f"Corpus {self.args.corpus}: "
                    f"{train_token_count:,} train tokens, {test_token_count:,} test tokens"
                ),
                Colors.CYAN,
            )
        )
        dataset = TextDataset(
            train_tokens=train_tokens,
            test_tokens=test_tokens,
            train_text=None,
            test_text=None,
            train_path=train_cache_path,
            test_path=test_cache_path,
        )

        self.dataset = dataset
        self.datasets_state.setdefault(self.args.corpus, dataset.state_dict())
        return (
            tokenizer,
            dataset,
            newline_token_id,
            boundary_blocklist,
            default_prompt_boundary,
            tokenizer_json,
        )

    def _ensure_active_corpus(
        self,
        payload: dict | None,
        model_path: pathlib.Path,
    ) -> bool:
        if self.args.corpus:
            return True
        corpus_name: str | None = None
        if isinstance(payload, dict):
            value = payload.get("corpus")
            if value:
                corpus_name = str(value)
        if not corpus_name:
            print(
                color_text(
                    (
                        f"Checkpoint {model_path} lacks corpus metadata. "
                        "Run 'grce.py corpus --set <name>' before training."
                    ),
                    Colors.RED,
                    bold=True,
                )
            )
            return False
        self.args.corpus = corpus_name
        return True

    def _build_fresh_corpus_state(self, name: str, vocab_size: int) -> dict[str, int]:
        data_dir = pathlib.Path(self.args.data)
        train_cache = data_dir / f"{name}_tokens_train_{vocab_size}.pt"
        test_cache = data_dir / f"{name}_tokens_test_{vocab_size}.pt"
        train_tokens = load_cached_tokens("train", train_cache)
        test_tokens = load_cached_tokens("test", test_cache)
        train_count = int(train_tokens.numel())
        test_count = int(test_tokens.numel())
        del train_tokens
        del test_tokens
        return {
            "train_cursor": 0,
            "test_cursor": 0,
            "train_cycles": 0,
            "test_cycles": 0,
            "train_count": train_count,
            "test_count": test_count,
        }

    def _token_cache_paths(self, corpus: str) -> tuple[pathlib.Path, pathlib.Path]:
        data_dir = pathlib.Path(self.args.data)
        train_cache = data_dir / f"{corpus}_tokens_train_{self.args.vocab_size}.pt"
        test_cache = data_dir / f"{corpus}_tokens_test_{self.args.vocab_size}.pt"
        return train_cache, test_cache

    def _corpus_tokens_available(self, corpus: str) -> bool:
        train_cache, test_cache = self._token_cache_paths(corpus)
        return train_cache.exists() and test_cache.exists()

    def _save_active_dataset_state(self) -> None:
        if self.dataset is None:
            return
        self.datasets_state[self.args.corpus] = self.dataset.state_dict()

    def _instantiate_dataset_for_corpus(self, corpus: str) -> TextDataset:
        train_cache, test_cache = self._token_cache_paths(corpus)
        train_tokens = load_cached_tokens("train", train_cache)
        test_tokens = load_cached_tokens("test", test_cache)
        dataset = TextDataset(
            train_tokens=train_tokens,
            test_tokens=test_tokens,
            train_text=None,
            test_text=None,
            train_path=train_cache,
            test_path=test_cache,
        )
        state = self.datasets_state.get(corpus)
        dataset.load_state(state)
        self.datasets_state[corpus] = dataset.state_dict()
        return dataset

    def _auto_advance_corpus_volume(self) -> bool:
        if self.tokenizer is None:
            return False
        current = self.args.corpus
        match = re.match(r"^(.*?)-(\d{4})$", current)
        if not match:
            return False
        prefix, digits = match.groups()
        next_idx = int(digits) + 1
        candidates: list[str] = [f"{prefix}-{next_idx:04d}"]
        fallback = f"{prefix}-0000"
        if fallback not in candidates:
            candidates.append(fallback)
        for candidate in candidates:
            if not self._corpus_tokens_available(candidate):
                continue
            self._save_active_dataset_state()
            new_dataset = self._instantiate_dataset_for_corpus(candidate)
            self.dataset = new_dataset
            self.args.corpus = candidate
            print(
                color_text(
                    f"Auto-switched to corpus {candidate}",
                    Colors.YELLOW,
                )
            )
            return True
        missing = ", ".join(candidates)
        raise FileNotFoundError(
            f"Unable to locate next corpus volume(s): {missing}. Add the pre-tokenized files or run 'corpus --init'."
        )

    def cli_prompts(
        self,
        tokenizer: GPT2TokenizerWrapper,
        model_path: pathlib.Path,
    ) -> int:
        target_path = Path(self.args.target) if self.args.target else model_path
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint {target_path} not found")
        payload = torch.load(target_path, map_location="cpu", weights_only=False)
        registry = PromptRegistry(tokenizer, payload.get("prompts"))
        changed = False
        if getattr(self.args, "reset", False):
            registry.reset_to_defaults()
            changed = True
        if getattr(self.args, "clear", False):
            registry.clear()
            changed = True
        removes = sorted(set(getattr(self.args, "remove", [])), reverse=True)
        if removes:
            registry.remove_indices(removes)
            changed = True
        add_pair = getattr(self.args, "add", None)
        if add_pair is not None:
            prompt_text, expected_text = add_pair
            registry.add_prompt(prompt_text, expected_text)
            changed = True
        if changed:
            payload["prompts"] = registry.serialize()
            torch.save(payload, target_path)
            print(color_text(f"Updated prompts in {target_path}", Colors.GREEN))
        show_list = self.args.list or not changed
        if show_list:
            entries = registry.ordered_entries()
            if not entries:
                print(color_text("No prompts stored in checkpoint", Colors.MAGENTA))
            else:
                print(color_text(f"Prompts in {target_path}:", Colors.CYAN))
                for idx, (prompt_text, entry) in enumerate(entries):
                    counts = color_text(
                        f"[argmax={entry.maxarg_count} sample={entry.sample_count}]",
                        Colors.YELLOW,
                    )
                    print(
                        color_text(f"#{idx}: ", Colors.CYAN)
                        + counts
                        + color_text(
                            f" prompt='{prompt_text}' expected='{entry.expected}'",
                            Colors.CYAN,
                        )
                    )
        return 0

    def cli_corpus(
        self,
        model_path: pathlib.Path,
        payload: dict | None,
    ) -> int:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Checkpoint {model_path} not found; run 'create' before managing corpora."
            )
        if payload is None:
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint payload must be a dictionary")
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError("Checkpoint lacks config metadata; re-run create to refresh it.")
        vocab_size = int(config.get("vocab_size", self.args.vocab_size))
        datasets = payload.get("datasets")
        if not isinstance(datasets, dict):
            datasets = {}
        add_name = getattr(self.args, "corpus_add", None)
        set_name = getattr(self.args, "corpus_set", None)
        updated = False

        def ensure_entry(name: str) -> dict[str, int]:
            nonlocal updated
            entry = datasets.get(name)
            if not isinstance(entry, dict):
                entry = {}
                datasets[name] = entry
            need_counts = "train_count" not in entry or "test_count" not in entry
            if not entry or need_counts:
                fresh = self._build_fresh_corpus_state(name, vocab_size)
                entry.setdefault("train_cursor", fresh["train_cursor"])
                entry.setdefault("test_cursor", fresh["test_cursor"])
                entry.setdefault("train_cycles", fresh["train_cycles"])
                entry.setdefault("test_cycles", fresh["test_cycles"])
                entry["train_count"] = fresh["train_count"]
                entry["test_count"] = fresh["test_count"]
                datasets[name] = entry
                updated = True
                print(color_text(f"Added corpus {name}", Colors.GREEN))
            return entry

        if add_name:
            ensure_entry(add_name)
        if set_name:
            ensure_entry(set_name)
            payload["corpus"] = set_name
            updated = True

        if updated:
            payload["datasets"] = datasets
            torch.save(payload, model_path)
            print(color_text(f"Saved corpus metadata to {model_path}", Colors.GREEN))

        current = payload.get("corpus") if isinstance(payload, dict) else None
        self._print_corpus_listing(datasets, current)
        return 0

    def _print_corpus_listing(self, datasets: dict, current: str | None) -> None:
        if not datasets:
            print(color_text("No corpora registered in checkpoint", Colors.YELLOW))
            return
        print(color_text("Registered corpora:", Colors.CYAN))
        for name in sorted(datasets):
            state = datasets.get(name) or {}
            prefix = "*" if current == name else "-"
            train_cursor = int(state.get("train_cursor", 0) or 0)
            test_cursor = int(state.get("test_cursor", 0) or 0)
            train_cycles = int(state.get("train_cycles", 0) or 0)
            test_cycles = int(state.get("test_cycles", 0) or 0)
            train_count = state.get("train_count")
            test_count = state.get("test_count")
            train_desc = f"train cursor {train_cursor:,} (cycles {train_cycles:,})"
            if isinstance(train_count, int):
                train_desc += f" / {train_count:,} tokens"
            test_desc = f"test cursor {test_cursor:,} (cycles {test_cycles:,})"
            if isinstance(test_count, int):
                test_desc += f" / {test_count:,} tokens"
            print(
                f"{prefix} {name}: {train_desc}; {test_desc}"
            )

    def big_fat_old_main(self) -> int:
        """Dispatch the CLI command selected by :func:`grce_cli_args`.

        Handles corpus management, training/reporting flow, and subcommands such
        as ``size``. When running training it constructs the model/tokenizer and
        calls :func:`train_model`.
        """

        ansi_file = None
        try:
            orig_stdout, orig_stderr, log_file = sys.stdout, sys.stderr, None

            tokenizer: GPT2TokenizerWrapper | None = None
            dataset: TextDataset | None = None
            newline_token_id: int | None = None
            boundary_blocklist: Sequence[int] | None = None
            default_prompt_boundary = False
            tokenizer_json: str | None = None

            model_dir = pathlib.Path(self.args.model)
            if not model_dir.exists():
                try:
                    model_dir.mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass

            if self.args.n_xctx > 0 and self.args.n_xctx % max(1, self.args.n_layer) != 0:
                raise ValueError("--n-xctx must be divisible by --n-layer")
            config = self.args
            model_tag = build_model_tag(config)
            for extra_tag in self.args.tag:
                cleaned = re.sub(r"[^0-9A-Za-z]+", "", extra_tag)
                if cleaned:
                    model_tag += f"_{cleaned}"
            model_path = getattr(self.args, "model_path_override", None)
            log_path = getattr(self.args, "log_path_override", None)
            if model_path is None or log_path is None:
                prefix = f"{self.args.name}_model_"
                model_path = model_dir / f"{prefix}{model_tag}.pt"
                log_path = model_dir / f"{prefix}{model_tag}.log"
                self.args.model_path_override = model_path
                self.args.log_path_override = log_path
            print(color_text(f"Model: {model_path}", Colors.CYAN))
            print(color_text(f"Logfile: {log_path}", Colors.BLUE))
            dataset_commands = {"train", "report", "test", "profile", "prompts"}
            requires_checkpoint = self.args.command in {"train", "report", "test", "profile", "prompts"}
            if self.args.command == "create" and model_path.exists():
                print(
                    color_text(
                        (
                            f"Checkpoint {model_path} already exists; delete it or pick a new --model directory."
                        ),
                        Colors.RED,
                        bold=True,
                    )
                )
                return 1
            if requires_checkpoint and not model_path.exists():
                print(
                    color_text(
                        (
                            f"Checkpoint {model_path} not found; run 'create' first to initialize it."
                        ),
                        Colors.RED,
                        bold=True,
                    )
                )
                return 1

            if self.args.command != "create" and self.args.tokenizer:
                print(
                    color_text(
                        "--tokenizer is only supported with the 'create' command; remove it and rerun.",
                        Colors.RED,
                        bold=True,
                    )
                )
                return 1

            payload = getattr(self.args, "checkpoint_payload_override", None)
            if payload is None and model_path.exists():
                payload = torch.load(
                    model_path,
                    map_location="cpu",
                    weights_only=False,
                )
                self.args.checkpoint_payload_override = payload

            if (
                isinstance(payload, dict)
                and payload.get("tokenizer_json")
                and not self.args.tokenizer_json_override
            ):
                self.args.tokenizer_json_override = payload.get("tokenizer_json")

            tokenizer_needed = self.args.command not in {"create", "corpus", "size"}
            if tokenizer_needed and not self.args.tokenizer_json_override:
                print(
                    color_text(
                        (
                            f"Checkpoint {model_path} lacks embedded tokenizer data. "
                            "Re-run 'create' to store the tokenizer snapshot."
                        ),
                        Colors.RED,
                        bold=True,
                    )
                )
                return 1

            if self.args.command == "corpus":
                return self.cli_corpus(model_path, payload)

            needs_dataset = self.args.command in dataset_commands
            if self.args.command == "create":
                (
                    tokenizer,
                    newline_token_id,
                    boundary_blocklist,
                    default_prompt_boundary,
                    tokenizer_json,
                ) = self._prepare_tokenizer_bundle(allow_files=True)
            elif needs_dataset:
                if not self._ensure_active_corpus(payload, model_path):
                    return 1
                (
                    tokenizer,
                    dataset,
                    newline_token_id,
                    boundary_blocklist,
                    default_prompt_boundary,
                    tokenizer_json,
                ) = self._prepare_corpus()
            else:
                (
                    tokenizer,
                    newline_token_id,
                    boundary_blocklist,
                    default_prompt_boundary,
                    tokenizer_json,
                ) = self._prepare_tokenizer_bundle(allow_files=False)

            if tokenizer is None:
                raise RuntimeError("Tokenizer initialization failed")

            if self.args.command == "prompts":
                return self.cli_prompts(tokenizer, model_path)
            sections = _append_summary_section(
                _expected_sections(config, config.block_size)
            )
            summary_items: list[dict] | None = None
            for key, _title, items in sections:
                if key == "summary":
                    summary_items = items
                    break
            if summary_items is None:
                summary_items = []
            summary_counts = {entry["label"]: entry["count"] for entry in summary_items}
            total_params = summary_counts.get("total", 0)
            embedding_params = summary_counts.get("embeddings", 0)
            non_emb_params = total_params - embedding_params
            print(
                f"Trainable model params: {total_params:,}; "
                f"excl. embeddings: {non_emb_params:,}"
            )
            tok_vecs = config.vocab_size
            pos_vecs = config.block_size
            emb_vectors = tok_vecs + pos_vecs
            emb_params = embedding_params
            print(
                f"Learned embedding vectors: {emb_vectors} "
                f"(token={tok_vecs}, position={pos_vecs}); params={emb_params:,}"
            )

            cmdline = " ".join(shlex.quote(arg) for arg in sys.argv)
            timestamp = datetime.now(timezone.utc).isoformat()
            log_file = log_path.open("a", encoding="utf-8")
            log_file.write(f"\n[{timestamp}] {cmdline}\n")
            log_file.flush()
            if not self.args.no_ansi:
                ansi_path = log_path.with_suffix(".ansi")
                ansi_file = ansi_path.open("a", encoding="utf-8")
                ansi_file.write(f"\n[{timestamp}] {cmdline}\n")
                ansi_file.flush()

            stdout_streams = [(orig_stdout, False), (log_file, True)]
            stderr_streams = [(orig_stderr, False), (log_file, True)]
            if ansi_file is not None:
                stdout_streams.append((ansi_file, False))
                stderr_streams.append((ansi_file, False))
            sys.stdout = Tee(*stdout_streams)
            sys.stderr = Tee(*stderr_streams)

            device = torch.device(self.args.device)
            try:
                prompt_tokens = tokenizer.encode(self.args.prompt)
            except KeyError as exc:  # pragma: no cover - user misconfiguration
                raise ValueError(
                    "Prompt contains characters outside the tokenizer vocabulary. "
                    "Choose a simpler prompt or extend the dataset."
                ) from exc
            try:
                prompt_tokens = prompt_tokens.unsqueeze(0).to(device)
            except (AssertionError, RuntimeError) as exc:
                message = str(exc)
                if "Torch not compiled with CUDA" in message and self.args.device != "cpu":
                    if not self.args.tiny:
                        raise RuntimeError(
                            "CUDA requested but not available; rerun with --device cpu or --tiny."
                        ) from exc
                    print(color_text("Torch not compiled with CUDA enabled; switching to CPU", Colors.RED, bold=True))
                    device = torch.device("cpu")
                    self.args.device = "cpu"
                    prompt_tokens = prompt_tokens.unsqueeze(0).to(device)
                else:
                    raise
            prompt_registry = PromptRegistry(tokenizer)

            try:
                model = GRCEGPT(config).to(device)
            except (AssertionError, RuntimeError) as exc:
                message = str(exc)
                if "Torch not compiled with CUDA" in message and self.args.device != "cpu":
                    if not self.args.tiny:
                        raise RuntimeError(
                            "CUDA requested but not available; rerun with --device cpu or --tiny."
                        ) from exc
                    print(color_text("Torch not compiled with CUDA enabled; switching to CPU", Colors.RED, bold=True))
                    device = torch.device("cpu")
                    model = GRCEGPT(config).to(device)
                else:
                    raise

            if self.args.torch_compile != "off":
                model = torch.compile(
                    model,
                    mode=self.args.torch_compile,
                    fullgraph=False,
                )

            total_steps = 0
            loss_history: List[Dict[str, float]] = []
            total_train_wall = 0.0
            payload = getattr(self.args, "checkpoint_payload_override", None)
            optimizer_state = None
            dataset_states: dict[str, dict[str, int]] = {}
            if payload is None and model_path.exists():
                if self.args.command == "create" and self.args.create_args.import_model:
                    raise ValueError(
                        "--import-model can only be used when no existing checkpoint is present"
                    )
                payload = torch.load(
                    model_path,
                    map_location=device,
                    weights_only=False,  # checkpoints also store dataset offsets/counters
                )
            if payload is not None:
                try:
                    if isinstance(payload, dict) and "model" in payload:
                        upgraded = upgrade_state_dict(payload["model"])
                        payload["model"] = upgraded
                        model.load_state_dict(upgraded)
                        total_steps = int(payload.get("total_steps", 0))
                        loss_history = list(payload.get("loss_history", []))
                        total_train_wall = float(payload.get("train_wall_seconds", 0.0))
                        prompt_registry = PromptRegistry(
                            tokenizer,
                            payload.get("prompts"),
                        )
                        if self.args.checkpoint_optimizer:
                            optimizer_state = payload.get("optimizer")
                        raw_datasets = payload.get("datasets")
                        if isinstance(raw_datasets, dict):
                            dataset_states = {
                                str(name): dict(state)
                                for name, state in raw_datasets.items()
                                if isinstance(state, dict)
                            }
                        elif "dataset" in payload and dataset is not None:
                            legacy_state = payload.get("dataset")
                            if isinstance(legacy_state, dict):
                                dataset.load_state(legacy_state)
                                legacy_name = payload.get("corpus") or self.args.corpus
                                dataset_states[legacy_name] = dataset.state_dict()
                        if "tokenizer_json" in payload:
                            self.args.tokenizer_json_override = payload.get("tokenizer_json")
                    else:
                        model.load_state_dict(upgrade_state_dict(payload))
                    print(color_text(f"Loaded existing model from {model_path}", Colors.YELLOW))
                    hours = total_train_wall / 3600.0
                    days = hours / 24.0
                    print(
                        color_text(
                            f"Total training so far: {total_steps} steps, {hours:.2f} hours ({days:.2f} days)",
                            Colors.YELLOW,
                        )
                    )
                except RuntimeError as err:
                    print(color_text("Checkpoint load failed (shape mismatch); starting fresh.", Colors.RED, bold=True))
                    print(color_text(str(err), Colors.RED))
            elif self.args.command == "create" and self.args.create_args.import_model:
                import_timer = Timer().start()
                if not self.args.create_args.import_model.exists():
                    raise FileNotFoundError(
                        f"Import checkpoint {self.args.create_args.import_model} not found"
                    )
                source_state, meta = load_checkpoint_payload(self.args.create_args.import_model, device)
                src_config = meta.get("config")
                if src_config is None:
                    raise ValueError(
                        "Imported checkpoint lacks config metadata; re-save it with the new format"
                    )
                if src_config.get("n_head") != config.n_head:
                    raise ValueError("Cannot import from a checkpoint with a different --n-head value")
                src_layers = src_config.get("n_layer")
                if src_layers is None:
                    src_layers = count_layers_from_state(source_state)
                print(color_text(f"Importing weights from {self.args.create_args.import_model}", Colors.GREEN))
                mapping = build_layer_mapping(
                    src_layers,
                    config.n_layer,
                    self.args.create_args.drop_layers,
                    self.args.create_args.add_layers,
                    allow_trim=self.args.create_args.trim_model,
                )
                apply_imported_state(
                    model,
                    source_state,
                    allow_trim=self.args.create_args.trim_model,
                    mapping=mapping,
                )
                total_steps = int(meta.get("total_steps", 0))
                total_train_wall = float(meta.get("train_wall_seconds", 0.0))
                loss_history = []
                write_timer = Timer().start()
                torch.save(
                    {
                        "model": model.state_dict(),
                        "total_steps": total_steps,
                        "loss_history": loss_history,
                        "config": asdict(args_to_model_geometry(self.args)),
                        "train_wall_seconds": total_train_wall,
                        "prompts": prompt_registry.serialize() if prompt_registry else None,
                        "tokenizer_json": tokenizer_json,
                    },
                    model_path,
                )
                print(
                    color_text(
                        f"[import] total steps: {total_steps}; time spent (wall/cpu/gpu): {import_timer.stop()}; writing model: {write_timer.stop()}",
                        Colors.CYAN,
                    )
                )
                return
            self.datasets_state = {
                name: dict(state) for name, state in dataset_states.items()
            }
            if dataset is not None and self.args.corpus:
                active_state = self.datasets_state.get(self.args.corpus)
                dataset.load_state(active_state)
                self.datasets_state[self.args.corpus] = dataset.state_dict()

            if self.args.command == "create":
                checkpoint_payload = {
                    "model": model.state_dict(),
                    "total_steps": 0,
                    "loss_history": [],
                    "config": asdict(args_to_model_geometry(self.args)),
                    "train_wall_seconds": 0.0,
                    "prompts": prompt_registry.serialize(),
                    "tokenizer_json": tokenizer_json,
                }
                torch.save(checkpoint_payload, model_path)
                print(
                    color_text(
                        f"Created new checkpoint at {model_path}; run 'corpus --set <name>' before training.",
                        Colors.GREEN,
                    )
                )
                return

            if self.args.command == "report":
                run_report_mode(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_tokens=prompt_tokens,
                    sample_len=self.args.generate,
                    count=self.args.report_count,
                    device=device,
                    suppress_newlines=self.args.no_newlines,
                    newline_token_id=newline_token_id,
                    default_prompt_boundary=default_prompt_boundary,
                    boundary_blocklist=boundary_blocklist,
                )
                return

            if self.args.command == "test":
                run_test_slice(
                    args=self.args,
                    dataset=dataset,
                    tokenizer=tokenizer,
                    model=model,
                    block_length=self.args.block_length,
                    start_pos=self.args.test_start,
                )
                return

            if self.args.command == "profile":
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
                if self.args.checkpoint_optimizer and optimizer_state:
                    try:
                        optimizer.load_state_dict(optimizer_state)
                    except Exception as err:  # pragma: no cover - logging
                        print(
                            color_text(
                                f"Warning: could not load optimizer state ({err}); starting fresh",
                                Colors.RED,
                            )
                        )
                run_profile_mode(
                    self.args,
                    dataset,
                    model,
                    optimizer,
                    block_length=self.args.block_length,
                    batch_size=self.args.batch_size,
                    device=device,
                    block_size=self.args.block_size,
                )
                return

            def build_optimizer() -> torch.optim.Optimizer:
                return torch.optim.AdamW(model.parameters(), lr=3e-4)

            shared_optimizer: torch.optim.Optimizer | None = None
            if not self.args.restart_optimizer:
                shared_optimizer = build_optimizer()
                if optimizer_state:
                    try:
                        shared_optimizer.load_state_dict(optimizer_state)
                    except Exception as err:  # pragma: no cover - logging
                        print(
                            color_text(
                                f"Warning: could not load optimizer state ({err}); starting fresh",
                                Colors.RED,
                            )
                        )
                optimizer_state = None
            else:
                optimizer_state = None

            acc_train = Timer()
            acc_eval = Timer()
            for cycle in range(1, self.args.cycles + 1):
                dataset = self.dataset
                cycle_wall = time.time()
                tags = ["GPT"]
                plus_tags: list[str] = []
                minus_tags: list[str] = []
                if self.args.n_grce > 0:
                    plus_tags.append("+GRCE")
                else:
                    minus_tags.append(" wo/GRCE")
                if self.args.n_xctx > 0:
                    plus_tags.append("+XCTX")
                else:
                    minus_tags.append(" wo/XCTX")
                label = "".join(tags + plus_tags + minus_tags)
                hours = total_train_wall / 3600.0
                days = hours / 24.0
                cycle_layouts = [
                    BatchLayout(
                        self.args.layout,
                        batch_size=self.args.batch_size,
                        block_size=self.args.block_length,
                    )
                    for _ in range(self.args.steps)
                ]
                step_token_spans = [layout.total_token_span() for layout in cycle_layouts]
                train_chars_cycle = sum(step_token_spans)
                if train_chars_cycle <= 0:
                    train_chars_cycle = (
                        (self.args.block_length + 1)
                        * self.args.batch_size
                        * self.args.steps
                    )
                train_chars_cycle = max(train_chars_cycle, 1)
                eval_steps = evaluation_step_indices(
                    self.args.steps,
                    self.args.eval_interval,
                )
                test_chars_cycle = sum(
                    step_token_spans[idx - 1]
                    for idx in eval_steps
                    if 1 <= idx <= len(step_token_spans)
                )
                if test_chars_cycle <= 0:
                    test_chars_cycle = (
                        max(step_token_spans)
                        if step_token_spans
                        else (self.args.block_length + 1) * self.args.batch_size
                    )
                test_chars_cycle = max(test_chars_cycle, 1)
                while True:
                    dataset = self.dataset
                    train_start = int(dataset.positions.get("train", 0))
                    train_wrapped = dataset.prepare_cycle("train", train_chars_cycle)
                    if train_wrapped and self._auto_advance_corpus_volume():
                        continue
                    test_start = int(dataset.positions.get("test", 0))
                    dataset.prepare_cycle("test", test_chars_cycle)
                    break

                train_chunk = dataset.chunks.get("train")
                test_chunk = dataset.chunks.get("test")

                def format_range(start: int, span: int) -> str:
                    if span <= 0:
                        return f"{start:,} - {start:,}"
                    end = start + span - 1
                    return f"{start:,} - {end:,}"

                train_span = int(train_chunk.size(0)) if train_chunk is not None else 0
                test_span = int(test_chunk.size(0)) if test_chunk is not None else 0
                train_range = format_range(train_start, train_span)
                test_range = format_range(test_start, test_span)
                print()
                pod_path = pathlib.Path(".podname")
                if pod_path.exists():
                    pod_label = pod_path.read_text(encoding="utf-8").strip()
                    if pod_label:
                        print(
                            color_text(
                                f"Running on remote pod {pod_label}.",
                                Colors.RED,
                                bold=True,
                            )
                        )
                print(color_text(f"Model: {model_path}", Colors.CYAN))
                print(
                    color_text(
                        f"Corpus {self.args.corpus}: train tokens {train_range}, test tokens {test_range}",
                        Colors.CYAN,
                    )
                )
                print(
                    color_text(
                        f"[{label}] Training Cycle {cycle}/{self.args.cycles}. "
                        f"Total training so far: {total_steps} steps, {hours:.2f} hours ({days:.2f} days)",
                        Colors.BLUE,
                    )
                )

                if self.args.restart_optimizer:
                    optimizer = build_optimizer()
                else:
                    if shared_optimizer is None:
                        shared_optimizer = build_optimizer()
                    optimizer = shared_optimizer

                (
                    total_steps,
                    updates,
                    train_timer,
                    eval_timer,
                ) = train_model(
                    self.args,
                    model,
                    dataset,
                    device,
                    self.args.steps,
                    self.args.block_length,
                    self.args.batch_size,
                    self.args.eval_interval,
                    total_steps,
                    optimizer,
                    prompt_tokens,
                    self.args.generate,
                    tokenizer,
                    suppress_newlines=self.args.no_newlines,
                    newline_token_id=newline_token_id,
                    prompt_registry=prompt_registry,
                    cycle_wall_start=cycle_wall,
                    base_wall_seconds=total_train_wall,
                    show_time=self.args.time,
                    default_prompt_boundary=default_prompt_boundary,
                    boundary_blocklist=boundary_blocklist,
                    show_train_loss_details=self.args.show_train_loss_details,
                    show_test_loss_details=self.args.show_test_loss_details,
                    prebuilt_layouts=cycle_layouts,
                )
                loss_history.extend(updates)
                pure_train = Timer().add(train_timer).sub(eval_timer)
                acc_train.add(pure_train)
                acc_eval.add(eval_timer)
                total_train_wall += train_timer.wall_secs
                self.datasets_state[self.args.corpus] = dataset.state_dict()
                if not self.args.skip_model_update:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "datasets": self.datasets_state,
                            "total_steps": total_steps,
                            "loss_history": loss_history,
                            "config": asdict(args_to_model_geometry(self.args)),
                            "train_wall_seconds": total_train_wall,
                            "corpus": self.args.corpus,
                            "prompts": prompt_registry.serialize() if prompt_registry else None,
                            "tokenizer_json": tokenizer_json,
                            **(
                                {"optimizer": optimizer.state_dict()}
                                if self.args.checkpoint_optimizer and not self.args.restart_optimizer
                                else {}
                            ),
                        },
                        model_path,
                    )
                cycle_part = color_text(f"[cycle {cycle} (wall/cpu/gpu)]", Colors.CYAN)
                train_part = color_text(f" train: {pure_train};", Colors.MAGENTA)
                eval_part = color_text(f" eval: {eval_timer};", Colors.GREEN)
                if self.args.skip_model_update:
                    updated_part = color_text(" model update skipped; flushing logs.", Colors.YELLOW)
                else:
                    updated_part = color_text(" model updated; flushing logs.", Colors.YELLOW)
                print(cycle_part + train_part + eval_part + updated_part)

                cumulative_part = color_text("[cumulative]", Colors.CYAN)
                cum_train_part = color_text(f" train: {acc_train};", Colors.MAGENTA)
                cum_eval_part = color_text(f" eval: {acc_eval};", Colors.GREEN)
                ratio_text = color_text(
                    f" train/eval: {acc_train.ratio(acc_eval)}",
                    Colors.CYAN,
                )
                print(cumulative_part + cum_train_part + cum_eval_part + ratio_text)
                if log_file is not None:
                    log_file.flush()
                if ansi_file is not None:
                    ansi_file.flush()

                if self.args.restart_optimizer:
                    # Drop the cycle-local optimizer before the next pass
                    optimizer = None

        except KeyboardInterrupt:
            if self.args.debug_interrupt:
                raise
            # traceback.print_exc()
            print(color_text("Interrupted by user; exiting cleanly.", Colors.RED, bold=True))
        except self.TimeoutAlarm:
            # traceback.print_exc()
            print(color_text("Timeout; exiting cleanly.", Colors.RED, bold=True))

        finally:
            self.cancel_timeout()
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            if log_file is not None:
                log_file.close()
            if ansi_file is not None:
                ansi_file.close()

        return 0

def grce_main(args: argparse.Namespace) -> int:
    """Primary entry point invoked from the CLI and unit tests."""

    # second entry point for "size" subcommand, now with
    # Torch imported; used only in 'size --check' mode
    if args.command == "size":
        assert args.size_check
        sys.exit(grce_cli_size(args))

    preprocess_runtime_args(args)

    torch.manual_seed(42)
    random.seed(time.time())

    # otherwise: run the big "default" main
    rt = Runtime(args)
    rt.start_timeout(args.timeout)
    return rt.big_fat_old_main()

if __name__ == "__main__":
    sys.exit(grce_main(cli_args))
