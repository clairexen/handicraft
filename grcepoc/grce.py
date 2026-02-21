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
    ("earth's satellite is the", " moon"),
    ("fire is hot and ice is", " cold"),
    ("ice is cold and fire is", " hot"),
    ("the opposite of up is", " down"),
    ("the sun rises in the", " east"),
    ("the sun sets in the", " west"),
    ("the color of a red apple is", " red"),
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

    vocab_size: int = 32000 # GPT-2 base supports ~50k merges.
    n_layer: int = 12       # GPT-2 base uses 12 layers.
    n_head: int = 12        # GPT-2 base uses 12 attention heads.
    n_embd: int = 768       # GPT-2 base uses 768 embedding dims.
    n_grce: int = 64        # Narrow GRCE context dims.
    n_xctx: int = 1536      # Wide XCTX context dims.
    n_rope_axis: int = 2    # Rotary axes per head (>=1 enables RoPE).
    n_rope_head: int = 0    # Number of heads using RoPE (0 disables RoPE).

MODEL_GEOMETRY_DEFAULTS = ModelGeometry()


@dataclass
class Defaults:
    """Default Settings (override with CLI args)"""

    vocab_size: int = MODEL_GEOMETRY_DEFAULTS.vocab_size
    block_size: int = 1024
    n_layer: int = MODEL_GEOMETRY_DEFAULTS.n_layer
    n_head: int = MODEL_GEOMETRY_DEFAULTS.n_head
    n_embd: int = MODEL_GEOMETRY_DEFAULTS.n_embd
    n_grce: int = MODEL_GEOMETRY_DEFAULTS.n_grce
    n_xctx: int = MODEL_GEOMETRY_DEFAULTS.n_xctx
    n_rope_axis: int = MODEL_GEOMETRY_DEFAULTS.n_rope_axis
    n_rope_head: int = MODEL_GEOMETRY_DEFAULTS.n_rope_head
    corpus: str | None = None
    steps: int = 100
    cycles: int = 100
    batch_size: int = 256
    layout: str = "2[*d]+2[*f],*[*1-2e=*1-4d=*1-4f=*1-2n],*[*1-2e=*1-4f=*1-4d=*1-2n]"
    eval_interval: int = 10
    dropout: float = 0.05
    detach_span: int = 0
    log_step_details: bool = False
    log_row_details: bool = False
    lr_base: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    lr_warmup_steps: int = 0
    lr_steady_steps: int = 0
    lr_linear_steps: int = 0
    lr_linear_min: float | None = None
    lr_cosine_steps: int = 0
    no_detach_ctx: bool = False
    prompt_no_prefix: bool = False

DEFAULTS = Defaults()


from argparse import Namespace as Args
GeometryLike = ModelGeometry | Args


FANCY_SPACE = "\u2423"  # Open Box symbol for visible spaces
FANCY_ENTER = "\u23CE " # Return symbol for visible newlines
PROMPT_PREFIX_TEXT = "\n\n"

def normalize_prompt(text: str) -> str:
    """Map placeholder characters back to literal spaces/newlines."""

    return text.replace(FANCY_SPACE, " ").replace(FANCY_ENTER, "\n"). \
            replace(FANCY_ENTER.replace(" ", "\n"), "\n")


def apply_prompt_prefix(text: str, *, enabled: bool) -> str:
    """Optionally prefix prompts before tokenization."""

    if not enabled:
        return text
    if text.startswith(PROMPT_PREFIX_TEXT):
        return text
    return f"{PROMPT_PREFIX_TEXT}{text}"


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
# * ``SEGMENTS`` are ``COLS``+``MODE`` tokens chained with ``=`` (e.g. ``16e=32d``).
#   ``>`` behaves like ``=`` but resets the KV cache before the following segment.
#   Legacy ``/`` separators are treated like ``=``.
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
from contextlib import contextmanager, nullcontext
import random
from typing import Iterator, List, Sequence


_MODE_ALIASES: dict[str, str] = {
    "e": "encode",
    "d": "decode",
    "r": "reverse",
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
    numerator: int | None = None
    denominator: int | None = None

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
        if "/" in token:
            if prefix is not None:
                raise LayoutParseError("Fractional sizes cannot use '*' or '+' prefixes")
            if "-" in token:
                raise LayoutParseError("Fractional sizes do not support ranges")
            num_text, den_text = token.split("/", 1)
            if not num_text or not den_text:
                raise LayoutParseError(f"Invalid fraction '{token}'")
            numerator = int(num_text)
            denominator = int(den_text)
            if numerator <= 0 or denominator <= 0:
                raise LayoutParseError("Fractional sizes require positive integers")
            return cls(
                minimum=0,
                maximum=0,
                expandable=False,
                constant_weight=False,
                cap=None,
                numerator=numerator,
                denominator=denominator,
            )
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

    def is_fraction(self) -> bool:
        return self.numerator is not None and self.denominator is not None

    def sample(self, rng: random.Random, base_size: int | None = None) -> int:
        if self.is_fraction():
            if base_size is None:
                raise ValueError("Fractional size requires a reference size")
            return (self.numerator * base_size) // self.denominator  # type: ignore
        if self.minimum == self.maximum:
            return self.minimum
        return rng.randint(self.minimum, self.maximum)


@dataclass
class SegmentSpec:
    size: CountSpec
    mode: str
    context_enabled: bool = True
    connector: str | None = None
    suppress_positional: bool = False


@dataclass(frozen=True)
class RowModifiers:
    detach_kv_cache: bool = False
    detach_span: int | None = None
    no_detach_ctx: bool = False
    train_transformer_only: bool = False
    train_recurrent_only: bool = False
    raw: str = ""

    def render(self) -> str:
        return self.raw


@dataclass
class RowSpec:
    count: CountSpec
    segments: list[SegmentSpec]
    modifiers: RowModifiers | None = None


@dataclass
class SegmentLayout:
    mode: str
    columns: int
    context_enabled: bool = True
    connector: str | None = None
    suppress_positional: bool = False


@dataclass
class BlockLayout:
    rows: int
    segments: list[SegmentLayout]
    modifiers: RowModifiers | None = None

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
    count_text, modifier_text = _split_row_count_and_modifiers(count_token)
    if not count_text:
        if modifier_text:
            count_text = "1"
        else:
            raise LayoutParseError("Row group missing row count")
    count_spec = CountSpec.parse(count_text)
    modifiers = _parse_row_modifiers(modifier_text)
    if not segments_body:
        raise LayoutParseError("Row group requires at least one segment")
    segment_tokens = _split_segments(segments_body)
    segments = [_parse_segment_spec(token, connector) for token, connector in segment_tokens]
    return RowSpec(count_spec, segments, modifiers)


def _split_segments(body: str) -> list[tuple[str, str | None]]:
    parts: list[tuple[str, str | None]] = []
    start = 0
    connector: str | None = None
    for index, ch in enumerate(body):
        if ch in "[]()":
            raise LayoutParseError("Unexpected bracket in segment string")
        if ch == "/":
            raise LayoutParseError("Use '=' between segments; '/' is reserved for fractions")
        if ch in "=>":
            token = body[start:index].strip()
            if token:
                parts.append((token, connector))
            connector = ch
            start = index + 1
    token = body[start:].strip()
    if token:
        parts.append((token, connector))
    return parts


def _parse_segment_spec(text: str, connector: str | None) -> SegmentSpec:
    text = text.strip()
    if not text:
        raise LayoutParseError("Empty segment definition")
    index = 0
    while index < len(text) and not text[index].isalpha():
        index += 1
    if index == len(text):
        raise LayoutParseError(f"Missing mode in segment '{text}'")
    size_token = text[:index]
    mode_token = text[index:]
    if not mode_token:
        raise LayoutParseError("Missing mode in segment")
    suppress_positional = False
    if mode_token[-1] in {"p", "P"}:
        suppress_positional = True
        mode_token = mode_token[:-1]
        if not mode_token:
            raise LayoutParseError("Positional modifier requires a base mode")
    mode_char = mode_token[0]
    context_enabled = mode_char.islower()
    mode_key = mode_char.lower()
    if mode_key not in _MODE_ALIASES:
        raise LayoutParseError(f"Unsupported mode '{mode_token}'")
    size_spec = CountSpec.parse(size_token or "1")
    if connector not in {None, "=", ">"}:
        raise LayoutParseError(f"Unsupported segment connector '{connector}'")
    return SegmentSpec(
        size_spec,
        _MODE_ALIASES[mode_key],
        context_enabled,
        connector,
        suppress_positional=suppress_positional,
    )


_ROW_COUNT_CHARS = set("0123456789+-*")


def _split_row_count_and_modifiers(token: str) -> tuple[str, str]:
    token = token.strip()
    if not token:
        return "", ""
    idx = 0
    while idx < len(token) and token[idx] in _ROW_COUNT_CHARS:
        idx += 1
    count_text = token[:idx]
    modifiers_text = token[idx:]
    return count_text, modifiers_text


def _parse_row_modifiers(text: str) -> RowModifiers | None:
    text = text.strip()
    if not text:
        return None
    idx = 0
    detach_kv = False
    detach_span: int | None = None
    no_detach_ctx = False
    train_transformer_only = False
    train_recurrent_only = False
    raw_parts: list[str] = []
    while idx < len(text):
        ch = text[idx]
        if ch == "k":
            detach_kv = True
            raw_parts.append("k")
            idx += 1
            continue
        if ch == "c":
            no_detach_ctx = True
            raw_parts.append("c")
            idx += 1
            continue
        if ch in {"T", "C"}:
            if ch == "T":
                if train_recurrent_only:
                    raise LayoutParseError("Row modifiers cannot include both 'T' and 'C'")
                train_transformer_only = True
            else:  # 'C'
                if train_transformer_only:
                    raise LayoutParseError("Row modifiers cannot include both 'T' and 'C'")
                train_recurrent_only = True
            raw_parts.append(ch)
            idx += 1
            continue
        if ch == "s":
            idx += 1
            start = idx
            while idx < len(text) and text[idx].isdigit():
                idx += 1
            digits = text[start:idx]
            span_value = int(digits) if digits else 1
            if span_value <= 0:
                raise LayoutParseError("s modifiers require a positive span")
            detach_span = span_value
            raw_parts.append("s" + digits)
            continue
        raise LayoutParseError(f"Unknown row modifier '{ch}' in '{text}'")
    raw = "".join(raw_parts)
    return RowModifiers(
        detach_kv_cache=detach_kv,
        detach_span=detach_span,
        no_detach_ctx=no_detach_ctx,
        train_transformer_only=train_transformer_only,
        train_recurrent_only=train_recurrent_only,
        raw=raw,
    )


@dataclass
class _CountAllocation:
    spec: CountSpec
    value: int
    fixed: bool = False

    def can_shrink(self) -> bool:
        if self.fixed:
            return False
        return self.value > self.spec.minimum

    def shrink(self) -> None:
        if not self.can_shrink():
            raise ValueError("Cannot shrink below minimum")
        self.value -= 1

    def can_expand(self) -> bool:
        if self.fixed:
            return False
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

    def _materialize_rows(self, specs: Sequence[RowSpec]) -> list[BlockLayout]:
        row_allocs: list[_CountAllocation] = []
        for spec in specs:
            value = spec.count.sample(self.rng, base_size=self.batch_size)
            row_allocs.append(
                _CountAllocation(spec.count, value, fixed=spec.count.is_fraction())
            )
        if not _shrink_until(self.batch_size, row_allocs, self.rng):
            min_rows = sum(item.spec.minimum for item in row_allocs)
            self.warnings.append(
                f"Row count lower bounds ({min_rows}) exceed batch size ({self.batch_size})"
            )
        else:
            _expand_until(self.batch_size, row_allocs, self.rng)
        rows: list[BlockLayout] = []
        for spec, allocation in zip(specs, row_allocs):
            segments = self._materialize_segments(spec.segments)
            rows.append(BlockLayout(allocation.value, segments, spec.modifiers))
        total_rows = sum(row.rows for row in rows)
        if total_rows > self.batch_size:
            self.warnings.append(
                f"Resolved layout uses {total_rows} rows which exceeds batch size {self.batch_size}"
            )
        return rows

    def _materialize_segments(self, specs: Sequence[SegmentSpec]) -> list[SegmentLayout]:
        allocations: list[_CountAllocation] = []
        for spec in specs:
            value = spec.size.sample(self.rng, base_size=self.block_size)
            allocations.append(
                _CountAllocation(spec.size, value, fixed=spec.size.is_fraction())
            )
        if not _shrink_until(self.block_size, allocations, self.rng):
            min_cols = sum(item.spec.minimum for item in allocations)
            self.warnings.append(
                f"Segment lower bounds ({min_cols}) exceed block size ({self.block_size})"
            )
        else:
            _expand_until(self.block_size, allocations, self.rng)
        segments = [
            SegmentLayout(
                spec.mode,
                allocation.value,
                spec.context_enabled,
                spec.connector,
                suppress_positional=spec.suppress_positional,
            )
            for spec, allocation in zip(specs, allocations)
        ]
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

    def serialize_rows(self, rows: Sequence[BlockLayout]) -> str:
        row_bits: list[str] = []
        for row in rows:
            segment_bits = []
            for idx, segment in enumerate(row.segments):
                letter = _MODE_LETTERS.get(segment.mode, segment.mode[0])
                if not segment.context_enabled:
                    letter = letter.upper()
                suffix = ""
                if segment.suppress_positional:
                    suffix = "P" if letter.isupper() else "p"
                bit = f"{segment.columns}{letter}{suffix}"
                if idx > 0:
                    connector = segment.connector or "="
                    bit = connector + bit
                segment_bits.append(bit)
            modifier_text = row.modifiers.render() if row.modifiers else ""
            row_bits.append(f"{row.rows}{modifier_text}[{''.join(segment_bits)}]")
        return "+".join(row_bits)

    def expanded_rows(self) -> list[list[SegmentLayout]]:
        """Return the per-row segments with rows fully expanded."""

        rows: list[list[SegmentLayout]] = []
        for group in self.rows:
            for _ in range(group.rows):
                rows.append(
                    [
                        SegmentLayout(seg.mode, seg.columns, seg.context_enabled, seg.connector)
                        for seg in group.segments
                    ]
                )
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
from typing import Any, Dict, List, Tuple, Sequence, Callable


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

    def raw_flag_value(flag: str) -> str | None:
        for idx, arg in enumerate(raw_cli_args):
            if arg == flag:
                if idx + 1 < len(raw_cli_args):
                    return raw_cli_args[idx + 1]
                return None
            if arg.startswith(f"{flag}="):
                return arg.split("=", 1)[1]
        return None
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
        help="Maximum sequence length consumed during train/eval",
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
        "--n-rope-axis",
        type=int,
        default=DEFAULTS.n_rope_axis,
        help=(
            "Number of rotary axes (per head) applied to Q/K; 0 disables RoPE"
        ),
    )
    model_group.add_argument(
        "--n-rope-head",
        type=int,
        default=DEFAULTS.n_rope_head,
        help=(
            "Limit rotary embeddings to the first N attention heads; 0 disables RoPE"
        ),
    )
    model_group.add_argument(
        "--tiny",
        action="store_true",
        help=(
            "Shortcut for --vocab-size 600 --batch-size 12 --block-size 6 --n-layer 3 --n-head 2 "
            "--n-embd 8 --n-grce 4 --n-xctx 9 --steps 2 --eval-interval 1"
        ),
    )

    training_group = parser.add_argument_group("Training schedule")
    training_group.add_argument("--steps", type=int, default=DEFAULTS.steps, help="Training steps per cycle")
    training_group.add_argument(
        "--cycles",
        type=str,
        default="+1",
        help="Repeat the full training/eval/update cycle N times (supports +N to extend).",
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
        "--log-row-details",
        action="store_true",
        default=DEFAULTS.log_row_details,
        help="Print per-row metrics and window spans (implies --log-step-details)",
    )
    training_group.add_argument(
        "--lr-base",
        type=float,
        default=DEFAULTS.lr_base,
        help="Base learning rate after warmup",
    )
    training_group.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULTS.weight_decay,
        help="AdamW weight decay applied to matrix weights",
    )
    training_group.add_argument(
        "--adam-beta1",
        type=float,
        default=DEFAULTS.adam_beta1,
        help="AdamW beta1 hyper-parameter",
    )
    training_group.add_argument(
        "--adam-beta2",
        type=float,
        default=DEFAULTS.adam_beta2,
        help="AdamW beta2 hyper-parameter",
    )
    training_group.add_argument(
        "--adam-eps",
        type=float,
        default=DEFAULTS.adam_eps,
        help="AdamW epsilon value",
    )
    training_group.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=DEFAULTS.lr_warmup_steps,
        help="Linear warmup steps (0 disables warmup)",
    )
    training_group.add_argument(
        "--lr-steady-steps",
        type=int,
        default=DEFAULTS.lr_steady_steps,
        help="Steps to hold the base LR before decays begin",
    )
    training_group.add_argument(
        "--lr-linear-steps",
        type=int,
        default=DEFAULTS.lr_linear_steps,
        help="Steps for the linear decay phase down to --lr-linear-min",
    )
    training_group.add_argument(
        "--lr-linear-min",
        type=float,
        default=DEFAULTS.lr_linear_min,
        help="Target LR at the end of the linear decay (default: 0.1 * --lr-base)",
    )
    training_group.add_argument(
        "--lr-cosine-steps",
        type=int,
        default=DEFAULTS.lr_cosine_steps,
        help="Steps for the final cosine decay from --lr-linear-min to zero",
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
        "--freeze-transformer",
        action="store_true",
        help="Freeze transformer stack parameters so only recurrent channels train",
    )
    training_group.add_argument(
        "--freeze-recurrent",
        action="store_true",
        help="Freeze recurrent (GRCE/XCTX) parameters so only the transformer trains",
    )
    training_group.add_argument(
        "--restart-optimizer",
        action="store_true",
        help="Reinitialize the optimizer at the beginning of every cycle",
    )
    checkpoint_group = training_group.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint-optimizer",
        dest="checkpoint_optimizer",
        action="store_true",
        help="Serialize optimizer state to checkpoints so runs can resume without momentum reset",
    )
    checkpoint_group.add_argument(
        "--no-checkpoint-optimizer",
        dest="checkpoint_optimizer",
        action="store_false",
        help="Disable optimizer state checkpointing",
    )
    parser.set_defaults(checkpoint_optimizer=True)
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
        "--no-prompt-prefix",
        action="store_true",
        default=DEFAULTS.prompt_no_prefix,
        help="Do not prepend blank lines to prompts before tokenization",
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
    train_parser.add_argument(
        "--json",
        dest="train_json",
        action="store_true",
        help="Write a JSON checkpoint without model weights alongside the .pt file",
    )

    try_parser = subparsers.add_parser(
        "try",
        help="Run the training loop without overwriting the checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    try_parser.set_defaults(command="try", skip_model_update=True)
    try_parser.add_argument(
        "--json",
        dest="train_json",
        action="store_true",
        help="Write a JSON checkpoint without model weights alongside the .pt file",
    )

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
        help="Inspect tokens from the test corpus or custom text",
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
    test_parser.add_argument(
        "text",
        nargs="*",
        help=(
            "Optional literal text to evaluate instead of pulling a --start slice from"
            " the test corpus (quote the text to preserve spaces)."
        ),
    )
    test_parser.add_argument(
        "--attn-map",
        action="store_true",
        help="Render per-layer attention weights for the final prediction",
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run a layout evaluation on a deterministic slice",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_parser.set_defaults(command="eval")
    eval_parser.add_argument(
        "--start",
        dest="eval_start",
        type=int,
        default=0,
        help="Cursor offset within the test corpus to begin evaluation (ignored when passing text)",
    )
    eval_parser.add_argument(
        "--rand",
        dest="eval_random",
        type=int,
        default=0,
        help="When >0, run evaluation N times on random offsets instead of using --start",
    )
    eval_parser.add_argument(
        "text",
        nargs="*",
        help=(
            "Optional literal text to evaluate instead of pulling a --start slice from"
            " the test corpus (quote the text to preserve spaces)."
        ),
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

    json_parser = subparsers.add_parser(
        "json",
        help="Export the current checkpoint metadata to JSON and exit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    json_parser.set_defaults(command="json")


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
        action="store_true",
        help="Register the listed corpora without changing their order",
    )
    corpus_parser.add_argument(
        "--set",
        dest="corpus_set",
        action="store_true",
        help="Register the listed corpora (if needed) and move them to the front",
    )
    corpus_parser.add_argument(
        "--reuse",
        dest="corpus_reuse",
        type=float,
        default=1.0,
        help="Multiplier applied to num_train_tokens when computing max_train_tokens",
    )
    corpus_parser.add_argument(
        "names",
        nargs="*",
        help="Corpus names consumed by --add/--set (supports shell brace expansion)",
    )

    reset_parser = subparsers.add_parser(
        "reset",
        help="Clear loss history and completed cycle counters in a checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    reset_parser.set_defaults(command="reset")


    # --------------------------------------------------------
    # Run the args parser

    args = parser.parse_args()
    if getattr(args, "freeze_transformer", False) and getattr(args, "freeze_recurrent", False):
        parser.error("--freeze-transformer and --freeze-recurrent cannot be used together")
    setattr(args, "_lr_steady_defined", flag_present("--lr-steady-steps"))
    if hasattr(args, "lr_linear_min") and args.lr_linear_min is None:
        args.lr_linear_min = args.lr_base * 0.1
    args.completed_cycles = 0
    args.corpus = None
    args._cycles_is_delta = False
    args._cycles_delta = 0

    raw_cycles = getattr(args, "cycles", "+1")
    if isinstance(raw_cycles, str) and raw_cycles.startswith("+"):
        digits = raw_cycles[1:]
        if not digits:
            parser.error("--cycles +N requires a numeric offset")
        try:
            delta = int(digits)
        except ValueError:
            parser.error(f"Invalid value for --cycles: {raw_cycles}")
        if delta < 0:
            parser.error("--cycles +N requires N >= 0")
        args._cycles_is_delta = True
        args._cycles_delta = delta
        args.cycles = delta
    else:
        try:
            args.cycles = int(raw_cycles)
        except (TypeError, ValueError):
            parser.error(f"--cycles must be an integer or +N, not {raw_cycles!r}")

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
        if not flag_present("--eval-interval"):
            args.eval_interval = 1

    if not flag_present("--n-rope-head"):
        args.n_rope_head = args.n_head

    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.log_row_details:
        args.log_step_details = True

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
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_grce=args.n_grce,
        n_xctx=args.n_xctx,
        n_rope_axis=args.n_rope_axis,
        n_rope_head=args.n_rope_head,
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
        ("R", "rotary axes", getattr(config, "n_rope_axis", 0)),
        ("RH", "rotary heads", getattr(config, "n_rope_head", 0)),
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


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of features by 90 degrees (used by RoPE)."""

    last_dim = tensor.size(-1)
    tensor = tensor.view(*tensor.shape[:-1], last_dim // 2, 2)
    first = tensor[..., 0]
    second = tensor[..., 1]
    rotated = torch.stack((-second, first), dim=-1)
    return rotated.reshape(*rotated.shape[:-2], last_dim)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query/key tensors."""

    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


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
    """Stores train/test tensors and samples random spans from each corpus."""

    train_tokens: torch.Tensor
    test_tokens: torch.Tensor
    train_text: str | None
    test_text: str | None
    train_path: pathlib.Path
    test_path: pathlib.Path

    def state_dict(self) -> dict[str, int]:
        """Compat shim for legacy checkpoints; no rolling state is tracked now."""

        return {}

    def load_state(self, state: dict | None) -> None:
        """No-op since datasets are sampled from the full corpus every time."""

        _ = state

    def _tokens_for_split(self, split: str) -> torch.Tensor:
        if split == "train":
            return self.train_tokens
        if split == "test":
            return self.test_tokens
        raise ValueError(f"Unknown split {split!r}")

    def sample_row_batch(
        self,
        split: str,
        columns: int,
        rows: int,
        device: torch.device,
        *,
        rng: random.Random | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, int]]]:
        if columns <= 0 or rows <= 0:
            raise ValueError("Row sampling requires positive columns and rows")
        tokens = self._tokens_for_split(split)
        total = int(tokens.numel())
        seq_span = columns + 1
        if seq_span <= 1:
            raise ValueError("Row sampling span must exceed 1 token")
        if total <= 0:
            raise ValueError(f"No tokens available for split {split!r}")
        if seq_span > total:
            raise ValueError(
                f"Requested span {seq_span} exceeds available {total} tokens in split {split}"
            )
        rng = rng or random
        windows: list[torch.Tensor] = []
        metadata: list[dict[str, int]] = []
        for row_idx in range(rows):
            start = rng.randint(0, total - 1)
            chunk, _ = self._slice_with_wrap(tokens, None, start, seq_span)
            windows.append(chunk)
            end = start + seq_span - 1
            metadata.append(
                {
                    "token_start": start,
                    "token_end": end % total,
                    "token_span": seq_span,
                    "wrapped": 1 if end >= total else 0,
                    "total_tokens": total,
                    "row": row_idx + 1,
                }
            )
        stacked = torch.stack(windows)
        x = stacked[:, :-1].contiguous().to(device)
        y = stacked[:, 1:].contiguous().to(device)
        return x, y, metadata

    def _slice_with_wrap(
        self,
        tokens: torch.Tensor,
        text: str | None,
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
        self.head_dim = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        rope_axes = max(0, int(getattr(config, "n_rope_axis", 0)))
        rope_heads = max(0, int(getattr(config, "n_rope_head", 0)))
        max_axes = max(0, self.head_dim // 2)
        rope_axes = min(rope_axes, max_axes)
        rope_heads = min(rope_heads, self.n_head)
        self.rope_axes = rope_axes
        self.rope_dim = rope_axes * 2
        self.rope_heads = rope_heads
        if self.rope_dim > 0 and rope_heads > 0:
            axis_scales = torch.tensor(
                [math.sqrt(idx + 1.0) for idx in range(rope_axes)],
                dtype=torch.float32,
            )
            self.register_buffer("rope_axis_scales", axis_scales, persistent=False)
            self.use_rope = True
        else:
            self.register_buffer("rope_axis_scales", None, persistent=False)
            self.use_rope = False

    def _rope_cache(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_rope or self.rope_axis_scales is None:
            raise RuntimeError("RoPE cache requested but rotary embeddings are disabled")
        axis_scales = self.rope_axis_scales.to(device=position_ids.device, dtype=position_ids.dtype)
        axis_angles = position_ids.float().unsqueeze(-1) * axis_scales
        angles = axis_angles.repeat_interleave(2, dim=-1).to(dtype)
        cos = torch.cos(angles).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(1)
        return cos, sin

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
        attention_capture: AttentionCapture | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        k_full = self.key(x)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v_full = self.value(x)
        k_local = k_full.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v_local = v_full.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            if position_ids is None:
                raise ValueError("position_ids are required when --n-rope-axis/--n-rope-head > 0")
            cos, sin = self._rope_cache(position_ids, q.dtype)
            q_slice = q[:, : self.rope_heads, :, : self.rope_dim]
            k_slice = k_local[:, : self.rope_heads, :, : self.rope_dim]
            q_rot, k_rot = _apply_rotary_pos_emb(q_slice, k_slice, cos, sin)
            q[:, : self.rope_heads, :, : self.rope_dim] = q_rot
            k_local[:, : self.rope_heads, :, : self.rope_dim] = k_rot

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
        scores = (q @ all_k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        block_mask: torch.Tensor | None = None
        if attn_mode not in {"encode", "decode", "reverse", "noattn"}:
            raise ValueError(f"Unknown attention mode: {attn_mode}")
        if attn_mode == "decode" and not full_attention:
            block_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
            )
        elif attn_mode == "reverse" and not full_attention:
            block_mask = torch.tril(
                torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1
            )
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
            k_full.view(B, T, self.n_head, self.head_dim),
            v_full.view(B, T, self.n_head, self.head_dim),
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
        if attention_capture is not None and attention_capture.columns:
            for col in attention_capture.columns:
                if 0 <= col < local_weights.size(2):
                    weights = local_weights[:, :, col, :]
                    attention_capture.record(
                        layer_idx or 0,
                        attention_capture.absolute_offset + col,
                        weights,
                    )
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
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, "LayerCache"]:
        if x.size(1) != 1:
            raise ValueError("Incremental attention expects a single-token sequence")
        B, T, C = x.shape
        k_full = self.key(x)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v_full = self.value(x)
        v = v_full.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k_new = k_full.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.use_rope:
            if position_ids is None:
                raise ValueError("position_ids are required when --n-rope-axis/--n-rope-head > 0")
            cos, sin = self._rope_cache(position_ids, q.dtype)
            q_slice = q[:, : self.rope_heads, :, : self.rope_dim]
            k_slice = k_new[:, : self.rope_heads, :, : self.rope_dim]
            q_rot, k_rot = _apply_rotary_pos_emb(q_slice, k_slice, cos, sin)
            q[:, : self.rope_heads, :, : self.rope_dim] = q_rot
            k_new[:, : self.rope_heads, :, : self.rope_dim] = k_rot
        key_append = k_new.squeeze(2).unsqueeze(2)
        value_append = v.squeeze(2).unsqueeze(2)
        if write_cache:
            cache.append(key_append, value_append)
            k, v = cache.tensors()
        else:
            k, v = key_append, value_append
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
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
        attention_capture: AttentionCapture | None = None,
        position_ids: torch.Tensor | None = None,
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
            attention_capture=attention_capture,
            position_ids=position_ids,
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
        position_ids: torch.Tensor | None = None,
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
            position_ids=position_ids,
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
        self.control_emb = nn.Embedding(3, config.n_embd, padding_idx=0)
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
        position_offsets: torch.Tensor | None = None,
        attention_capture: AttentionCapture | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        rows, cols, _ = x.shape
        if position_ids is None:
            base = torch.arange(cols, device=x.device, dtype=torch.long)
            position_ids = base.unsqueeze(0).expand(rows, cols)
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
                attention_capture=attention_capture,
                position_ids=position_ids,
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
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list]:
        output, samples, kv_out = self.core.forward_grid(
            x,
            xctx_bias_list_in=xctx_bias_list_in,
            grce_bias_list_in=grce_bias_list_in,
            kv_cache_list_in=kv_cache_list_in,
            mode=mode,
            qh_query_callback=qh_query_callback,
            position_ids=position_ids,
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
        self.detach_ctx_enabled_default = not getattr(config, "no_detach_ctx", False)
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
        detach_ctx_enabled: bool | None = None,
        detach_span_override: int | None = None,
    ) -> torch.Tensor:
        if self.disabled:
            return grce_state
        span_value = self.detach_span if detach_span_override is None else max(0, detach_span_override)
        should_detach = detach_samples or (
            span_value > 0 and (position % span_value) == 0
        )
        state_detach_enabled = (
            self.detach_ctx_enabled_default if detach_ctx_enabled is None else detach_ctx_enabled
        )
        if should_detach and state_detach_enabled and grce_state is not None:
            grce_state = grce_state.detach()
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
        return self.output_norm(grce_state + mlp_out)

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
        self.detach_ctx_enabled_default = not getattr(config, "no_detach_ctx", False)
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
        detach_ctx_enabled: bool | None = None,
        detach_span_override: int | None = None,
    ) -> torch.Tensor:
        if self.disabled:
            return xctx_state
        span_value = self.detach_span if detach_span_override is None else max(0, detach_span_override)
        should_detach = detach_samples or (
            span_value > 0 and (position % span_value) == 0
        )
        state_detach_enabled = (
            self.detach_ctx_enabled_default if detach_ctx_enabled is None else detach_ctx_enabled
        )
        if should_detach and state_detach_enabled and xctx_state is not None:
            xctx_state = xctx_state.detach()
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


class TransformerStackColumn:
    """Evaluates a single column with optional GRCE/XCTX bias shims."""

    def __init__(
        self,
        core: "TransformerStackCore",
        grce_module: TransformerGRCE | None,
        xctx_module: TransformerXCTX | None,
    ) -> None:
        self.core = core
        self.grce = grce_module
        self.xctx = xctx_module

    @staticmethod
    def _collect_biases(
        bias_list: Sequence[torch.Tensor] | None,
        column_index: int,
    ) -> list[torch.Tensor]:
        if not bias_list:
            return []
        collected: list[torch.Tensor] = []
        for bias in bias_list:
            if bias is None or bias.size(1) == 0:
                continue
            if bias.size(1) == 1:
                collected.append(bias)
            elif column_index < bias.size(1):
                collected.append(bias[:, column_index : column_index + 1, :, :])
        return collected

    def forward(
        self,
        column_input: torch.Tensor,
        *,
        column_index: int,
        grce_state: torch.Tensor | None,
        xctx_state: torch.Tensor | None,
        grce_bias_list_in: Sequence[torch.Tensor] | None,
        xctx_bias_list_in: Sequence[torch.Tensor] | None,
        kv_sources: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]] | None] | None,
        qh_query_callback=None,
        detach_internal_kv_cache: bool = False,
        attention_capture: "AttentionCapture" | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        grce_biases = self._collect_biases(grce_bias_list_in, column_index)
        xctx_biases = self._collect_biases(xctx_bias_list_in, column_index)
        if self.grce is not None and grce_state is not None:
            grce_biases.append(self.grce.bias_forward(grce_state))
        if self.xctx is not None and xctx_state is not None:
            xctx_biases.append(self.xctx.bias_forward(xctx_state))
        column_output, samples, kv_pairs = self.core.forward_grid(
            column_input,
            xctx_bias_list_in=xctx_biases,
            grce_bias_list_in=grce_biases,
            kv_cache_list_in=kv_sources,
            mode="decode",
            qh_query_callback=qh_query_callback,
            attention_capture=attention_capture,
            position_ids=position_ids,
        )
        if detach_internal_kv_cache:
            column_output = column_output.detach()
            samples = [sample.detach() for sample in samples]
        return column_output, samples, kv_pairs


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
        self.column = TransformerStackColumn(self.core, self.grce, self.xctx)

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
        context_detach_span: int | None = None,
        context_detach_enabled: bool | None = None,
        attention_capture: AttentionCapture | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list]:
        if mode not in {"forward", "encode", "decode", "reverse", "noattn"}:
            raise ValueError(f"Unknown TransformerStackSequence mode: {mode}")
        rows, cols, _ = x.shape
        device = x.device
        dtype = x.dtype
        outputs: list[torch.Tensor] = []
        grce_state = self._ensure_state(self.grce, grce_in, rows, device, dtype)
        xctx_state = self._ensure_state(self.xctx, xctx_in, rows, device, dtype)
        if position_ids is None:
            base = torch.arange(cols, device=device, dtype=torch.long)
            position_ids = base.unsqueeze(0).expand(rows, cols)
        if mode in {"encode", "decode", "reverse"}:
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
                context_detach_span=context_detach_span,
                context_detach_enabled=context_detach_enabled,
                attention_capture=attention_capture,
                position_ids=position_ids,
            )
        use_internal_cache = mode != "noattn"
        base_sources = list(kv_cache_list_in or [])
        kv_history: list[list[tuple[torch.Tensor, torch.Tensor] | None]] = []
        kv_storage: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if detach_internal_kv_cache:
            kv_storage = self._allocate_detached_kv_storage(rows, cols, device, dtype)
        for col in range(cols):
            column_input = x[:, col : col + 1, :]
            column_kv_sources: list[Sequence[tuple[torch.Tensor, torch.Tensor] | None]] = []
            if base_sources:
                column_kv_sources.extend(base_sources)
            if detach_internal_kv_cache and kv_storage is not None and col > 0:
                column_kv_sources.append(self._detached_kv_prefix(kv_storage, col))
            elif (not detach_internal_kv_cache) and use_internal_cache and kv_history:
                column_kv_sources.extend(kv_history)
            kv_sources_arg = column_kv_sources if column_kv_sources else None
            column_capture = None
            if attention_capture is not None:
                column_capture = attention_capture.subset(col, 1)
            column_output, samples, kv_pairs = self.column.forward(
                column_input,
                column_index=col,
                grce_state=grce_state,
                xctx_state=xctx_state,
                grce_bias_list_in=grce_bias_list_in,
                xctx_bias_list_in=xctx_bias_list_in,
                kv_sources=kv_sources_arg,
                qh_query_callback=qh_query_callback,
                detach_internal_kv_cache=detach_internal_kv_cache,
                attention_capture=column_capture,
                position_ids=position_ids[:, col : col + 1],
            )
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
                    detach_ctx_enabled=context_detach_enabled,
                    detach_span_override=context_detach_span,
                )
            if self.xctx is not None and xctx_state is not None:
                if detach_xctx_span > 0 and (col % detach_xctx_span) == 0:
                    xctx_state = xctx_state.detach()
                xctx_state = self.xctx.sample_forward(
                    xctx_state,
                    samples,
                    col,
                    detach_samples=detach_samples,
                    detach_ctx_enabled=context_detach_enabled,
                    detach_span_override=context_detach_span,
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
        context_detach_span: int | None,
        context_detach_enabled: bool | None,
        attention_capture: AttentionCapture | None,
        position_ids: torch.Tensor,
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
            attention_capture=attention_capture,
            position_ids=position_ids,
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
                    detach_ctx_enabled=context_detach_enabled,
                    detach_span_override=context_detach_span,
                )
            if self.xctx is not None and xctx_state is not None:
                if detach_xctx_span > 0 and (col % detach_xctx_span) == 0:
                    xctx_state = xctx_state.detach()
                xctx_state = self.xctx.sample_forward(
                    xctx_state,
                    slice_samples,
                    col,
                    detach_samples=detach_samples,
                    detach_ctx_enabled=context_detach_enabled,
                    detach_span_override=context_detach_span,
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
        self._transformer_params = tuple(self.core.parameters())
        recurrent_params: list[nn.Parameter] = []
        for module in self.context_channels:
            recurrent_params.extend(list(module.parameters()))
        self._recurrent_params = tuple(recurrent_params)

    def _position_ids(
        self,
        length: int,
        batch_size: int,
        device: torch.device,
        position_offsets: torch.Tensor | None,
    ) -> torch.Tensor:
        base = torch.arange(length, device=device, dtype=torch.long).unsqueeze(0).expand(
            batch_size, -1
        )
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
        tok = self.core.tok_emb(idx)
        x = self.core.drop(tok)
        context_info: dict[str, torch.Tensor] | None = None
        if mode in {"forward", "noattn", "encode"}:
            sequence_output, grce_out, xctx_out, _ = self.stack_sequence.forward(
                x,
                mode=mode,
                position_ids=pos_idx,
            )
            hidden = sequence_output
            context_info = {}
            if grce_out is not None:
                context_info["grce"] = grce_out
            if xctx_out is not None:
                context_info["xctx"] = xctx_out
            if not context_info:
                context_info = None
        elif mode in {"decode", "reverse"}:
            decode_output, _, _ = self.stack_grid.forward(
                x,
                mode=mode,
                position_ids=pos_idx,
            )
            hidden = decode_output
        else:
            raise ValueError(f"Unknown forward_autoreg mode: {mode}")
        logits = self.core.head(self.core.ln_f(hidden))
        return logits, None, context_info

    @contextmanager
    def grad_scope(
        self,
        *,
        transformer: bool | None = None,
        recurrent: bool | None = None,
    ) -> Iterator[None]:
        toggled: list[tuple[nn.Parameter, bool]] = []
        try:
            self._apply_grad_toggle(self._transformer_params, transformer, toggled)
            self._apply_grad_toggle(self._recurrent_params, recurrent, toggled)
            yield
        finally:
            for param, prev in reversed(toggled):
                param.requires_grad_(prev)

    @staticmethod
    def _apply_grad_toggle(
        params: Sequence[nn.Parameter],
        enabled: bool | None,
        toggled: list[tuple[nn.Parameter, bool]],
    ) -> None:
        if enabled is None or not params:
            return
        for param in params:
            if param.requires_grad != enabled:
                toggled.append((param, param.requires_grad))
                param.requires_grad_(enabled)


def build_model_tag(config: GeometryLike) -> str:
    """Build the filename tag used by ``train``/``create`` checkpoints."""

    tag = (
        f"v{config.vocab_size}_emb{config.n_embd}_"
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


BATCH_MODES: tuple[str, ...] = ("reverse", "encode", "decode", "forward", "noattn")


ROW_METRIC_HIST_KEYS = list(BATCH_MODES)
ROW_METRIC_LOG_KEYS = list(BATCH_MODES)
ROW_METRIC_LOG_GROUP = {"reverse", "forward"}


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
    dataset: TextDataset,
    split: str,
    rows: Sequence[BlockLayout],
    device: torch.device,
    *,
    collect_mode_metrics: bool,
    position_shift: int = 0,
    rng: random.Random | None = None,
    row_serializer: Callable[[Sequence[BlockLayout]], str] | None = None,
) -> tuple[LayoutPassResult, float, list[dict[str, object]]]:
    start_time = time.time()
    rng = rng or random
    mode_loss_sums = {mode: 0.0 for mode in BATCH_MODES}
    mode_token_counts = {mode: 0 for mode in BATCH_MODES}
    total_loss_sum: torch.Tensor | None = None
    total_tokens = 0
    row_details: list[dict[str, object]] = []
    for group in rows:
        row_count = int(group.rows)
        if row_count <= 0:
            continue
        cols_total = group.total_columns()
        if cols_total <= 0:
            continue
        modifiers = group.modifiers
        detach_span_override = (
            modifiers.detach_span if modifiers and modifiers.detach_span is not None else None
        )
        row_detach_kv_cache = args.detach_kv_cache or (
            modifiers.detach_kv_cache if modifiers else False
        )
        context_detach_override = False if (modifiers and modifiers.no_detach_ctx) else None
        layout_text = row_serializer([group]) if row_serializer else ""
        training_scope = nullcontext()
        if torch.is_grad_enabled():
            transformer_flag: bool | None = None
            recurrent_flag: bool | None = None
            if getattr(args, "freeze_transformer", False):
                transformer_flag = False
            if getattr(args, "freeze_recurrent", False):
                recurrent_flag = False
            if modifiers:
                if modifiers.train_transformer_only:
                    transformer_flag = True
                    recurrent_flag = False
                elif modifiers.train_recurrent_only:
                    transformer_flag = False
                    recurrent_flag = True
            if transformer_flag is not None or recurrent_flag is not None:
                training_scope = model.grad_scope(
                    transformer=transformer_flag,
                    recurrent=recurrent_flag,
                )
        with training_scope:
            try:
                xb, yb, metadata = dataset.sample_row_batch(
                    split,
                    cols_total,
                    row_count,
                    device,
                    rng=rng,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unable to sample {row_count} rows with {cols_total} columns from the corpus"
                ) from exc
            pos_offsets = None
            if position_shift:
                pos_offsets = torch.full((row_count,), position_shift, dtype=torch.long, device=device)
        token_components, position_ids = _embedding_components_with_offsets(
            model,
            xb,
            pos_offsets,
        )
        future_token_components, _ = _embedding_components_with_offsets(
            model,
            yb,
            pos_offsets,
        )
        control_ids = torch.zeros((row_count, cols_total), dtype=torch.long, device=device)
        row_entries: list[dict[str, object]] = []
        for idx in range(row_count):
            meta = metadata[idx] if idx < len(metadata) else {}
            row_entries.append(
                {
                    "layout": layout_text,
                    "row": int(meta.get("row", idx + 1)),
                    "token_start": int(meta.get("token_start", 0)),
                    "token_end": int(meta.get("token_end", 0)),
                    "token_span": int(meta.get("token_span", 0)),
                    "wrapped": bool(meta.get("wrapped", 0)),
                    "total_tokens": int(meta.get("total_tokens", 0)),
                    "loss_sum": 0.0,
                    "token_count": 0,
                }
            )
            cursor = 0
            kv_chain: list[list[tuple[torch.Tensor, torch.Tensor]] | None] = []
            grce_state = None
            xctx_state = None
            for segment in group.segments:
                cols = int(segment.columns)
                if cols <= 0:
                    continue
                mode = segment.mode
                connector = getattr(segment, "connector", None)
                if connector == ">":
                    kv_chain = []
                start = cursor
                end = cursor + cols
                if mode == "reverse":
                    control_ids[:, start:end] = CONTROL_PREDICT_PREV
                elif mode in {"decode", "forward", "noattn"}:
                    control_ids[:, start:end] = CONTROL_PREDICT_NEXT
                elif mode == "encode" and end > start:
                    control_ids[:, start:end] = CONTROL_NONE
                    control_ids[:, end - 1 : end] = CONTROL_PREDICT_NEXT
                control_slice = control_ids[:, start:end].clone()
                control_embed = model.core.control_emb(control_slice)
                if mode == "reverse":
                    token_source = future_token_components
                    chunk_target = xb[:, start:end]
                else:
                    token_source = token_components
                    chunk_target = yb[:, start:end]
                token_slice = token_source[:, start:end, :]
                pos_id_slice = position_ids[:, start:end]
                if segment.suppress_positional:
                    pos_id_slice = pos_id_slice.new_zeros(pos_id_slice.shape)
                chunk_input = _compose_chunk_embeddings(
                    model.core.drop,
                    token_slice,
                    control_slice=control_embed,
                )
                kv_sources = None if mode == "noattn" else (kv_chain if kv_chain else None)
                prev_grce_state = grce_state
                prev_xctx_state = xctx_state
                chunk_output, grce_state, xctx_state, kv_out = model.stack_sequence.forward(
                    chunk_input,
                    grce_in=grce_state,
                    xctx_in=xctx_state,
                    kv_cache_list_in=kv_sources,
                    mode=mode,
                    position_ids=pos_id_slice,
                    detach_internal_kv_cache=row_detach_kv_cache,
                    context_detach_span=detach_span_override,
                    context_detach_enabled=context_detach_override,
                )
                if not getattr(segment, "context_enabled", True):
                    grce_state = prev_grce_state
                    xctx_state = prev_xctx_state
                logits = model.core.head(model.core.ln_f(chunk_output))
                (
                    loss_sum,
                    token_count,
                    row_loss_sums,
                    row_token_counts,
                ) = loss_sum_token_count_with_rows(
                    logits,
                    chunk_target,
                    last_only=(mode == "encode"),
                )
            if token_count > 0:
                total_tokens += token_count
                total_loss_sum = loss_sum if total_loss_sum is None else total_loss_sum + loss_sum
                metric_key = mode
                if collect_mode_metrics and metric_key in mode_loss_sums:
                    mode_loss_sums[metric_key] += float(loss_sum.detach().item())
                    mode_token_counts[metric_key] += token_count
                    if row_loss_sums is not None and row_token_counts is not None:
                        loss_values = row_loss_sums.detach().cpu().tolist()
                        token_values = row_token_counts.detach().cpu().tolist()
                        for idx, entry in enumerate(row_entries):
                            if idx >= len(loss_values):
                                break
                            entry["loss_sum"] = entry.get("loss_sum", 0.0) + float(loss_values[idx])
                            entry["token_count"] = int(entry.get("token_count", 0)) + int(token_values[idx])
                if mode != "noattn":
                    kv_chain.append(kv_out)
                cursor += cols
            row_details.extend(row_entries)
    result = LayoutPassResult(total_loss_sum, total_tokens, mode_loss_sums, mode_token_counts)
    return result, time.time() - start_time, row_details


def train_layout_batch(
    args: Args,
    model: GRCEGPT,
    dataset: TextDataset,
    layout: BatchLayout,
    device: torch.device,
    grad_hook: Callable[[int, int], None] | None = None,
    position_shift: int = 0,
) -> tuple[torch.Tensor, int, list[tuple[int, float, float, str]], dict[str, object]]:
    step_span = layout.total_token_span()
    if step_span <= 0:
        raise ValueError("Layout produced zero tokens for training step")
    window_rng = random.Random()
    total_loss_sum: torch.Tensor | None = None
    total_tokens = 0
    micro_logs: list[tuple[int, float, float, str]] = []
    detail_entries: list[dict[str, object]] = []
    layout_text = layout.serialize()
    for index, batch in enumerate(layout.micro_batches, start=1):
        micro_span = sum(row.token_span() for row in batch)
        if micro_span <= 0:
            micro_logs.append((index, 0.0, 0.0, layout.serialize_rows(batch)))
            continue
        rows_text = layout.serialize_rows(batch)
        micro_detail = {
            "index": index,
            "token_span": micro_span,
            "row_count": sum(max(0, row.rows) for row in batch),
            "rows_text": rows_text,
            "layout": layout_text,
        }
        try:
            result, fwd_time, row_details = _run_microbatch_pass(
                args,
                model,
                dataset,
                "train",
                batch,
                device,
                collect_mode_metrics=False,
                position_shift=position_shift,
                rng=window_rng,
                row_serializer=layout.serialize_rows,
            )
        except torch.OutOfMemoryError as exc:
            if not hasattr(exc, "microbatch_detail"):
                exc.microbatch_detail = micro_detail
            raise
        for entry in row_details:
            copy = dict(entry)
            copy["micro_index"] = index
            detail_entries.append(copy)
        if result.total_loss_sum is None or result.total_tokens <= 0:
            micro_logs.append((index, fwd_time, 0.0, rows_text))
            continue
        bwd_start = time.time()
        result.total_loss_sum.backward()
        if grad_hook is not None:
            grad_hook(index, micro_span)
        bwd_time = time.time() - bwd_start
        micro_logs.append((index, fwd_time, bwd_time, rows_text))
        total_loss_sum = (
            result.total_loss_sum
            if total_loss_sum is None
            else total_loss_sum + result.total_loss_sum
        )
        total_tokens += result.total_tokens
    if total_loss_sum is None:
        raise RuntimeError("Layout batch produced no tokens")
    meta_entry: dict[str, object] = {"rows": detail_entries}
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
    for batch_rows in layout.micro_batches:
        micro_span = sum(row.token_span() for row in batch_rows)
        if micro_span <= 0:
            continue
        result, _, _ = _run_microbatch_pass(
            args,
            model,
            dataset,
            split,
            batch_rows,
            device,
            collect_mode_metrics=True,
            rng=window_rng,
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
    return _loss_sum_token_count_internal(logits, targets, last_only, False)[0:2]


def loss_sum_token_count_with_rows(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    last_only: bool = False,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    return _loss_sum_token_count_internal(logits, targets, last_only, True)


def _loss_sum_token_count_internal(
    logits: torch.Tensor,
    targets: torch.Tensor,
    last_only: bool,
    return_rows: bool,
) -> tuple[torch.Tensor, int, torch.Tensor | None, torch.Tensor | None]:
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
    if not return_rows:
        return loss_sum, token_count, None, None
    row_loss_sums = (per_token * valid_mask).sum(dim=1)
    row_token_counts = valid_mask.sum(dim=1)
    return loss_sum, token_count, row_loss_sums, row_token_counts


CONTROL_NONE = 0
CONTROL_PREDICT_NEXT = 1
CONTROL_PREDICT_PREV = 2


def _embedding_components_with_offsets(
    model: GRCEGPT,
    token_batch: torch.Tensor,
    position_offsets: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = token_batch.shape
    pos_idx = model._position_ids(seq_len, batch_size, token_batch.device, position_offsets)
    token_emb = model.core.tok_emb(token_batch)
    return token_emb, pos_idx


def _compose_chunk_embeddings(
    dropout_layer: nn.Dropout,
    token_slice: torch.Tensor,
    *,
    control_slice: torch.Tensor | None = None,
) -> torch.Tensor:
    base = token_slice
    if control_slice is not None:
        base = base + control_slice
    return dropout_layer(base)


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


def _optimizer_param_groups(module: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith("bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups if groups else [{"params": module.parameters(), "weight_decay": weight_decay}]


def _scheduled_lr(
    base_lr: float,
    warmup_steps: int,
    steady_steps: int,
    steady_defined: bool,
    linear_steps: int,
    linear_min: float,
    cosine_steps: int,
    total_run_steps: int,
    step_index: int,
) -> float:
    base_lr = max(0.0, base_lr)
    if base_lr <= 0.0:
        return 0.0
    warmup_steps = max(0, warmup_steps)
    steady_steps = max(0, steady_steps)
    linear_steps = max(0, linear_steps)
    cosine_steps = max(0, cosine_steps)
    linear_min = max(0.0, min(linear_min, base_lr))
    step = max(0, step_index)
    if not steady_defined:
        remaining = total_run_steps - warmup_steps - linear_steps - cosine_steps
        steady_steps = max(0, remaining)

    if warmup_steps > 0:
        if step < warmup_steps:
            return base_lr * float(step + 1) / float(warmup_steps)
        step -= warmup_steps

    if steady_steps > 0:
        if step < steady_steps:
            return base_lr
        step -= steady_steps

    current_lr = base_lr
    if linear_steps > 0:
        if step < linear_steps:
            ratio = step / max(1, linear_steps)
            return current_lr + (linear_min - current_lr) * ratio
        step -= linear_steps
        current_lr = linear_min

    if cosine_steps > 0:
        if step < cosine_steps:
            progress = step / max(1, cosine_steps)
            return 0.5 * current_lr * (1.0 + math.cos(math.pi * progress))
        return 0.0

    return current_lr


def atomic_torch_save(payload: dict, target_path: pathlib.Path) -> None:
    tmp_path = target_path.with_suffix(target_path.suffix + "_")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tmp_path)
    tmp_path.replace(target_path)


def _jsonify_checkpoint_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify_checkpoint_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify_checkpoint_value(item) for item in value]
    return str(value)


def write_checkpoint_json(payload: dict[str, Any], target_path: pathlib.Path) -> None:
    sanitized = {key: _jsonify_checkpoint_value(val) for key, val in payload.items()}
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n", encoding="utf-8")




def train_model(
    args: Args,
    model: GRCEGPT,
    dataset: TextDataset,
    device: torch.device,
    steps: int,
    batch_size: int,
    eval_interval: int,
    start_step: int,
    optimizer: torch.optim.Optimizer,
    sample_prompt: torch.Tensor,
    sample_chars: int,
    total_train_tokens_start: int,
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
) -> Tuple[int, List[Dict[str, float]], float, float, int]:
    """Run the main training loop for a cycle."""

    if optimizer is None:
        raise ValueError("train_model requires an initialized optimizer instance")
    total_steps = start_step
    history_updates: List[Dict[str, float]] = []
    printed_header = False
    eval_interval = max(1, int(eval_interval))
    loop_timer = Timer().start()
    eval_timer = Timer()
    train_tokens_used = 0
    base_total_train_tokens = int(total_train_tokens_start)
    run_total_steps = max(1, args.steps * args.cycles)
    current_lr = args.lr_base

    long_loss_header = " ".join([""] + [f"{': ' if key in ROW_METRIC_LOG_GROUP else ''}{key}" for key in ROW_METRIC_LOG_KEYS])

    header_columns: List[Tuple[str, str]] = []
    if show_time:
        header_columns.append(("time", Colors.BLUE))
    header_columns.append(("step", Colors.CYAN))
    header_columns.append(
        (
            "train" + (long_loss_header if show_train_loss_details else ""),
            Colors.MAGENTA,
        )
    )
    header_columns.append(
        (
            "test" + (long_loss_header if show_test_loss_details else ""),
            Colors.GREEN,
        )
    )
    header_columns.append(("lr", Colors.YELLOW))
    header_line = " | ".join(color_text(label, color) for label, color in header_columns)
    print(header_line + " |")

    oom_retries = 0
    oom_retry_limit = 10
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
            layout = BatchLayout(args.layout, batch_size=batch_size, block_size=args.block_size)
        layout_serialized = layout.serialize()
        layout_span = layout.total_token_span()
        _log_layout_warnings(args, layout)
        current_step_index = total_steps + 1
        step_wall_start = time.time()
        try:
            current_lr = _scheduled_lr(
                args.lr_base,
                args.lr_warmup_steps,
                args.lr_steady_steps,
                getattr(args, "_lr_steady_defined", False),
                args.lr_linear_steps,
                args.lr_linear_min,
                args.lr_cosine_steps,
                run_total_steps,
                max(0, total_steps),
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)
            micro_grad_norms.clear()
            def _record_micro_grad(micro_idx: int, micro_tokens: int) -> None:
                raw_norm = _grad_norm(model)
                norm = raw_norm / max(1, micro_tokens)
                if args.log_grad_norms:
                    micro_grad_norms.append((micro_idx, micro_tokens, raw_norm, norm))
                if args.grad_summary:
                    cycle_micro_norms.append(norm)
            position_shift = 0
            total_loss_sum, total_tokens, micro_logs, window_detail = train_layout_batch(
                args,
                model,
                dataset,
                layout,
                device,
                grad_hook=_record_micro_grad if need_grad_tracking else None,
                position_shift=position_shift,
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
            train_tokens_used += total_tokens
        except torch.OutOfMemoryError as oom_err:
            oom_retries += 1
            line_parts: List[str] = []
            if show_time:
                timestamp = time.strftime("%H:%M", time.localtime())
                line_parts.append(color_text(timestamp, Colors.BLUE))
            line_parts.append(color_text(f"{total_steps}", Colors.CYAN))
            detail_note = ""
            detail = getattr(oom_err, "microbatch_detail", None)
            if isinstance(detail, dict):
                pieces: list[str] = []
                micro_idx = detail.get("index")
                if micro_idx is not None:
                    pieces.append(f"micro {micro_idx}")
                span = detail.get("token_span")
                if span:
                    pieces.append(f"span {span}")
                row_count = detail.get("row_count")
                if row_count:
                    pieces.append(f"rows {row_count}")
                rows_text = detail.get("rows_text")
                if rows_text:
                    pieces.append(rows_text)
                if pieces:
                    detail_note = " (" + "; ".join(str(piece) for piece in pieces if piece) + ")"
            line_parts.append(color_text(
                f"OOM (retry {oom_retries}/{oom_retry_limit}) during layout {layout_serialized}{detail_note}; "
                "refreshing layout and retrying",
                Colors.YELLOW,
            ))
            line = " | ".join(line_parts)
            print(line)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            if oom_retries >= oom_retry_limit:
                raise
            max_span = layout_span
            replacement = layout
            attempts = 0
            while True:
                candidate = BatchLayout(args.layout, batch_size=batch_size, block_size=args.block_size)
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
            row_meta = window_detail.get("rows") if isinstance(window_detail, dict) else None
            row_count_desc = f" ({len(row_meta)} row windows)" if isinstance(row_meta, list) else ""
            header = f"Step {current_step_index}: u-batch | fwd | bwd | layout{row_count_desc}"
            print(color_text(header, Colors.BLUE))
            total_fwd = 0.0
            total_bwd = 0.0
            for idx, fwd_time, bwd_time, rows_str in micro_logs:
                total_fwd += fwd_time
                total_bwd += bwd_time
                line = (
                    f"  {idx} | {fwd_time:.2f}s | {bwd_time:.2f}s | {rows_str or layout_serialized}"
                )
                print(color_text(line, Colors.BLUE))
            other_time = max(0.0, step_wall - (total_fwd + total_bwd + opt_duration))
            summary = f"  {opt_duration:.2f}s optimize, {other_time:.2f}s other"
            print(color_text(summary, Colors.BLUE))
            if args.log_row_details and isinstance(row_meta, list) and row_meta:
                print(color_text("  per-row details:", Colors.BLUE))
                for detail in row_meta:
                    micro_idx = detail.get("micro_index")
                    layout_text = detail.get("layout") or ""
                    start = int(detail.get("token_start", 0))
                    end = int(detail.get("token_end", 0))
                    span = int(detail.get("token_span", 0))
                    wrapped = " wrap" if detail.get("wrapped") else ""
                    token_count = int(detail.get("token_count", 0))
                    loss_sum = float(detail.get("loss_sum", 0.0))
                    avg_loss = loss_sum / token_count if token_count > 0 else None
                    row_no = detail.get("row")
                    row_line = (
                        f"    micro {micro_idx} row {row_no}: tokens {start}-{end}"
                        f" (span {span}{wrapped})"
                    )
                    if avg_loss is not None:
                        row_line += f" | avg loss {avg_loss:.4f}"
                    if layout_text:
                        row_line += f" | {layout_text}"
                    print(color_text(row_line, Colors.BLUE))
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
                prefixed_prompt = apply_prompt_prefix(
                    picked_prompt,
                    enabled=getattr(args, "prompt_prefix", True),
                )
                prompt_tokens = tokenizer.encode(prefixed_prompt).unsqueeze(0).to(device)
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
        line_parts.append(color_text(f"{current_lr:.2e}", Colors.YELLOW))
        line = " | ".join(line_parts) + " | " + sample_render
        print(line)

        eval_now = time.time()
        cycle_wall_elapsed = max(0.0, eval_now - cycle_wall_start)
        total_wall_seconds = base_wall_seconds + cycle_wall_elapsed

        current_total_train_tokens = base_total_train_tokens + train_tokens_used
        record = {
            "step": total_steps,
            "train_loss": float(eval_metrics["train"].metrics.get("target", 0.0) or 0.0),
            "test_loss": float(eval_metrics["test"].metrics.get("target", 0.0) or 0.0),
            "train_wall_seconds": float(total_wall_seconds),
            "unix_time": float(eval_now),
            "train_tokens": total_tokens,
            "total_train_tokens": int(current_total_train_tokens),
        }
        record["corpus"] = args.corpus
        record["batch_layout"] = layout_serialized
        record["learning_rate"] = current_lr
        metric_keys = list(ROW_METRIC_HIST_KEYS)
        for key in metric_keys:
            train_val = eval_metrics["train"].metrics.get(key)
            test_val = eval_metrics["test"].metrics.get(key)
            if train_val is not None:
                record[f"train_loss_{key}"] = float(train_val)
            if test_val is not None:
                record[f"test_loss_{key}"] = float(test_val)
        history_updates.append(record)

    if args.grad_summary:
        summary = color_text("[grad norms]", Colors.CYAN) + " "
        if cycle_micro_norms:
            summary += color_text(
                f"micro min {min(cycle_micro_norms):.4f} max {max(cycle_micro_norms):.4f} avg {sum(cycle_micro_norms)/len(cycle_micro_norms):.4f}; ",
                Colors.MAGENTA,
            )
        if cycle_step_norms:
            summary += color_text(
                f"steps min {min(cycle_step_norms):.4f} max {max(cycle_step_norms):.4f} avg {sum(cycle_step_norms)/len(cycle_step_norms):.4f}",
                Colors.GREEN,
            )
        summary += color_text(
            f"; processed {train_tokens_used:,} tokens",
            Colors.BLUE,
        )
        print(summary)

    return total_steps, history_updates, loop_timer.stop(), eval_timer, train_tokens_used


def run_profile_mode(
    args: Args,
    dataset: TextDataset,
    model: GRCEGPT,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    device: torch.device,
    ) -> None:
    """Warm up once, profile a second training step, and report CUDA stats."""

    try:
        from torch.profiler import ProfilerActivity, profile, record_function
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "torch.profiler is unavailable; upgrade to PyTorch 1.8+ to use 'profile'."
        ) from exc

    profile_layout = BatchLayout(args.layout, batch_size=batch_size, block_size=args.block_size)
    _log_layout_warnings(args, profile_layout)

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


def _format_token_fragment(tokenizer: GPT2TokenizerWrapper, token_id: int) -> str:
    piece = tokenizer.decode_one(int(token_id))
    if not piece:
        return "<∅>"
    piece = piece.replace(" ", FANCY_SPACE)
    replacement = FANCY_ENTER if "\n" in piece else None
    if replacement is not None:
        piece = piece.replace("\n", replacement)
    return piece


def _row_description(layout: BatchLayout, row: BlockLayout) -> str:
    return layout.serialize_rows([BlockLayout(1, row.segments, row.modifiers)])


@dataclass
class AttentionCapture:
    columns: set[int]
    storage: dict[int, dict[int, torch.Tensor]]
    absolute_offset: int = 0

    def subset(self, start: int, length: int) -> "AttentionCapture | None":
        if not self.columns:
            return None
        local = {
            col - start
            for col in self.columns
            if start <= col < start + length
        }
        if not local:
            return None
        return AttentionCapture(
            columns=local,
            storage=self.storage,
            absolute_offset=self.absolute_offset + start,
        )

    def record(self, layer_idx: int, column_index: int, weights: torch.Tensor) -> None:
        layer_store = self.storage.setdefault(layer_idx, {})
        absolute_col = self.absolute_offset + column_index
        layer_store[absolute_col] = weights.squeeze(0).detach().cpu()


@dataclass
class RowEvalResult:
    logits: torch.Tensor
    supervision_mask: torch.Tensor
    column_modes: list[str]
    mode_loss_sums: dict[str, float]
    mode_token_counts: dict[str, int]
    total_loss_sum: float
    total_tokens: int
    target_ids: torch.Tensor
    attention_maps: dict[int, dict[int, torch.Tensor]] | None = None
    block_attentions: list["BlockAttention"] | None = None


@dataclass
class BlockAttention:
    mode: str
    tokens: list[int]
    target_token: int
    layer_weights: dict[int, torch.Tensor]


def _evaluate_row_block(
    args: Args,
    model: GRCEGPT,
    row: BlockLayout,
    token_components: torch.Tensor,
    future_token_components: torch.Tensor,
    source_ids: torch.Tensor,
    targets: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    input_tokens: torch.Tensor,
    capture_columns: set[int] | None = None,
) -> RowEvalResult:
    device = token_components.device
    eval_block_length = token_components.size(1)
    column_modes: list[str] = [""] * eval_block_length
    supervision_mask = torch.ones(eval_block_length, dtype=torch.bool, device=device)
    mode_loss_sums = {mode: 0.0 for mode in BATCH_MODES}
    mode_token_counts = {mode: 0 for mode in BATCH_MODES}
    total_loss = 0.0
    total_tokens = 0
    logits_buffer: list[torch.Tensor] = []
    attention_storage: dict[int, dict[int, torch.Tensor]] = {}
    base_capture = None
    if capture_columns:
        base_capture = AttentionCapture(
            columns=set(capture_columns),
            storage=attention_storage,
            absolute_offset=0,
        )
    block_attentions: list[BlockAttention] = []
    cursor = 0
    kv_chain: list[list[tuple[torch.Tensor, torch.Tensor]] | None] = []
    grce_state = None
    xctx_state = None
    modifiers = row.modifiers
    detach_span_override = (
        modifiers.detach_span if modifiers and modifiers.detach_span is not None else None
    )
    row_detach_kv_cache = args.detach_kv_cache or (
        modifiers.detach_kv_cache if modifiers else False
    )
    context_detach_override = False if (modifiers and modifiers.no_detach_ctx) else None
    control_ids = torch.zeros((1, eval_block_length), dtype=torch.long, device=device)
    target_ids = torch.zeros_like(targets)
    for segment in row.segments:
        segment_start = cursor
        cols = int(segment.columns)
        if cols <= 0:
            continue
        if cursor + cols > eval_block_length:
            raise ValueError(
                "Layout segment exceeds available token columns during evaluation"
            )
        connector = getattr(segment, "connector", None)
        if connector == ">":
            kv_chain = []
        start = cursor
        end = cursor + cols
        if segment.mode == "reverse":
            control_ids[:, start:end] = CONTROL_PREDICT_PREV
        elif segment.mode in {"decode", "forward", "noattn"}:
            control_ids[:, start:end] = CONTROL_PREDICT_NEXT
        elif segment.mode == "encode" and end > start:
            control_ids[:, start:end] = CONTROL_NONE
            control_ids[:, end - 1 : end] = CONTROL_PREDICT_NEXT
        control_slice = control_ids[:, start:end]
        control_embed = model.core.control_emb(control_slice)
        if segment.mode == "reverse":
            token_source = future_token_components
            chunk_target = source_ids[:, start:end]
        else:
            token_source = token_components
            chunk_target = targets[:, start:end]
        token_slice = token_source[:, start:end, :]
        pos_id_slice = position_ids[:, start:end]
        if segment.suppress_positional:
            pos_id_slice = pos_id_slice.new_zeros(pos_id_slice.shape)
        chunk_input = _compose_chunk_embeddings(
            model.core.drop,
            token_slice,
            control_slice=control_embed,
        )
        chunk_capture = None
        if base_capture is not None:
            chunk_capture = base_capture.subset(cursor, cols)
        kv_sources = None if segment.mode == "noattn" else (kv_chain if kv_chain else None)
        prev_grce_state = grce_state
        prev_xctx_state = xctx_state
        chunk_output, grce_state, xctx_state, kv_out = model.stack_sequence.forward(
            chunk_input,
            grce_in=grce_state,
            xctx_in=xctx_state,
            kv_cache_list_in=kv_sources,
            mode=segment.mode,
            position_ids=pos_id_slice,
            detach_internal_kv_cache=row_detach_kv_cache,
            context_detach_span=detach_span_override,
            context_detach_enabled=context_detach_override,
            attention_capture=chunk_capture,
        )
        if not segment.context_enabled:
            grce_state = prev_grce_state
            xctx_state = prev_xctx_state
        target_ids[:, start:end] = chunk_target
        logits = model.core.head(model.core.ln_f(chunk_output))
        logits_buffer.append(logits)
        loss_sum, token_count = loss_sum_and_token_count(
            logits,
            chunk_target,
            last_only=(segment.mode == "encode"),
        )
        if token_count > 0:
            loss_value = float(loss_sum.detach().item())
            metric_key = segment.mode
            if metric_key in mode_loss_sums:
                mode_loss_sums[metric_key] += loss_value
                mode_token_counts[metric_key] += token_count
            total_loss += loss_value
            total_tokens += token_count
        if segment.mode != "noattn":
            kv_chain.append(kv_out)
        for local_idx in range(cols):
            idx = cursor + local_idx
            column_modes[idx] = segment.mode
            if segment.mode == "encode" and local_idx < cols - 1:
                supervision_mask[idx] = False
        if chunk_capture is not None and segment.mode in {"decode", "reverse", "encode"}:
            abs_col = chunk_capture.absolute_offset + (cols - 1)
            layer_weights: dict[int, torch.Tensor] = {}
            for layer_idx, store in attention_storage.items():
                weights = store.get(abs_col)
                if weights is not None:
                    layer_weights[layer_idx] = weights[:, -cols:].contiguous()
            if layer_weights:
                token_slice = input_tokens[segment_start : segment_start + cols]
                block_tokens = token_slice.view(-1).tolist()
                target_tensor = target_ids[0, segment_start + cols - 1]
                target_token = int(target_tensor.item())
                block_attentions.append(
                    BlockAttention(
                        mode=segment.mode,
                        tokens=block_tokens,
                        target_token=target_token,
                        layer_weights=layer_weights,
                    )
                )
        cursor += cols
    if cursor != eval_block_length:
        raise ValueError("Layout columns do not match the evaluated token span")
    if not logits_buffer:
        raise ValueError("Row block produced no segments during evaluation")
    row_logits = torch.cat(logits_buffer, dim=1)
    return RowEvalResult(
        logits=row_logits,
        supervision_mask=supervision_mask,
        column_modes=column_modes,
        mode_loss_sums=mode_loss_sums,
        mode_token_counts=mode_token_counts,
        total_loss_sum=total_loss,
        total_tokens=total_tokens,
        target_ids=target_ids,
        attention_maps=attention_storage if capture_columns else None,
        block_attentions=block_attentions,
    )


def _prepare_eval_tokens(
    args: Args,
    dataset: TextDataset,
    tokenizer: GPT2TokenizerWrapper,
    *,
    start_pos: int,
    custom_text: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str, str]:
    if custom_text:
        provided = tokenizer.encode(custom_text)
        if provided.numel() < 2:
            raise ValueError("Custom text must produce at least two tokens for evaluation")
        if provided.numel() - 1 > args.block_size:
            raise ValueError(
                "Custom text exceeds the configured --block-size; shorten the text or increase --block-size."
            )
        context_tokens = provided
        source_label = "custom text"
    else:
        span = args.block_size + 1
        if span <= 1:
            raise ValueError("--block-size must be >= 1 for evaluation")
        context_tokens = dataset.looped_slice("test", start_pos, span)
        source_label = f"test split offset {start_pos}"
    if context_tokens.numel() < 2:
        raise ValueError("Not enough tokens collected for evaluation")
    inputs = context_tokens[:-1]
    targets = context_tokens[1:]
    eval_block_length = inputs.numel()
    pretty_text = tokenizer.decode_pretty(args, context_tokens)
    return context_tokens, inputs, targets, eval_block_length, source_label, pretty_text


def run_test_slice(
    args: Args,
    dataset: TextDataset,
    tokenizer: GPT2TokenizerWrapper,
    model: GRCEGPT,
    start_pos: int,
    *,
    custom_text: str | None = None,
) -> None:
    """Run the layout on either a corpus slice or custom text and log per-token stats."""

    model_device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    with torch.no_grad():
        (
            context_tokens,
            inputs,
            targets,
            eval_block_length,
            source_label,
            pretty_text,
        ) = _prepare_eval_tokens(
            args,
            dataset,
            tokenizer,
            start_pos=start_pos,
            custom_text=custom_text,
        )

        layout = BatchLayout(args.layout, batch_size=args.batch_size, block_size=eval_block_length)
        _log_layout_warnings(args, layout)

        print(color_text(f"Evaluating layout '{args.layout}' on {source_label}:", Colors.CYAN))
        print(pretty_text)
        print(color_text(
            f"Sequence tokens: {eval_block_length} inputs (context) + 1 target tail", Colors.CYAN
        ))

        xb = inputs.unsqueeze(0).to(model_device)
        yb = targets.unsqueeze(0).to(model_device)
        token_components, position_ids = _embedding_components_with_offsets(
            model,
            xb,
            None,
        )
        future_token_components, _ = _embedding_components_with_offsets(
            model,
            yb,
            None,
        )
        vocab_size = model.config.vocab_size

        column_tokens = inputs.clone()

        for row_idx, row in enumerate(layout.rows, start=1):
            row_desc = _row_description(layout, row)
            print(color_text(f"Row block #{row_idx}: {row_desc}", Colors.YELLOW))

            capture_columns = None
            if getattr(args, "attn_map", False):
                capture_columns = {eval_block_length - 1}
            try:
                row_result = _evaluate_row_block(
                    args,
                    model,
                    row,
                    token_components,
                    future_token_components,
                    xb,
                    yb,
                    position_ids,
                    input_tokens=column_tokens,
                    capture_columns=capture_columns,
                )
            except ValueError as exc:
                print(color_text(f"  (error evaluating row: {exc})", Colors.RED))
                continue

            row_logits = row_result.logits
            log_probs = torch.log_softmax(row_logits, dim=-1)
            target_ids = row_result.target_ids
            gathered = torch.gather(
                log_probs,
                dim=-1,
                index=target_ids.unsqueeze(-1),
            ).squeeze(-1)
            per_token_loss = (-gathered).squeeze(0)

            top_k = min(5, vocab_size)
            top_logp, top_indices = torch.topk(log_probs, k=top_k, dim=-1)
            top_probs = top_logp.exp()

            losses_cpu = per_token_loss.cpu().tolist()
            mask_cpu = row_result.supervision_mask.cpu().tolist()
            inputs_cpu = column_tokens.tolist()
            targets_cpu = row_result.target_ids.squeeze(0).cpu().tolist()
            modes_cpu = row_result.column_modes
            top_indices_cpu = top_indices.squeeze(0).cpu().tolist()
            top_probs_cpu = top_probs.squeeze(0).cpu().tolist()

            idx_width = 4
            token_width = max(
                len(_format_token_fragment(tokenizer, tok)) for tok in inputs_cpu + targets_cpu
            )
            pad = " " * 4
            for col in range(eval_block_length):
                token_text = _format_token_fragment(tokenizer, inputs_cpu[col])
                loss_value = losses_cpu[col] if mask_cpu[col] else None
                ranking: list[str] = []
                for idx, prob in zip(top_indices_cpu[col], top_probs_cpu[col]):
                    token_piece = _format_token_fragment(tokenizer, idx)
                    ranking.append(f"{token_piece} ({prob * 100:.1f}%)")
                loss_display = f"{loss_value:7.3f}" if loss_value is not None else "   --  "
                idx_text = f"{col:4d}"
                print(
                    f"{pad}{idx_text} | {token_text:<{token_width}} | {loss_display} | {', '.join(ranking)}"
                )

            summary_target = _format_token_fragment(tokenizer, targets_cpu[-1])
            row_total_loss = row_result.total_loss_sum
            if row_result.total_tokens > 0:
                row_avg_loss_text = f"{row_total_loss / row_result.total_tokens:7.3f}"
            else:
                row_avg_loss_text = "   --  "
            summary_label = "*" * idx_width
            print(
                f"{pad}{summary_label} | {summary_target:<{token_width}} | {row_avg_loss_text} nats/token"
            )

            if getattr(args, "attn_map", False) and row_result.block_attentions:
                for block in row_result.block_attentions:
                    _print_attention_heatmap(
                        tokenizer,
                        block,
                        pad,
                        token_width,
                        model.config.n_layer,
                    )

    if was_training:
        model.train()


def _print_attention_heatmap(
    tokenizer: GPT2TokenizerWrapper,
    block: BlockAttention,
    pad: str,
    token_width: int,
    n_layers: int,
) -> None:
    layers = [idx for idx in range(n_layers) if idx in block.layer_weights]
    if not layers:
        return
    target_text = _format_token_fragment(tokenizer, block.target_token)
    print(color_text(
        f"Attention heatmap for predicting {target_text} ({block.mode} block):",
        Colors.CYAN,
    ))
    header = f"{pad}{'':4s} | {'':<{token_width}} | "
    header += " ".join(f"L_{idx}".rjust(4) for idx in layers)
    print(header)
    for idx, token in enumerate(block.tokens):
        token_text = _format_token_fragment(tokenizer, token)
        row_text = f"{pad}{idx:4d} | {token_text:<{token_width}} |"
        for layer_idx in layers:
            weights = block.layer_weights.get(layer_idx)
            if weights is None or idx >= weights.size(1):
                row_text += " ----"
                continue
            head_weights = weights[:, idx]
            digits = []
            for head_weight in head_weights:
                scaled = min(9, int(round(float(head_weight.item()) * 9)))
                digits.append(str(scaled))
            row_text += f" {''.join(digits):>4}"
        print(row_text)

def _format_eval_metric_value(key: str, value: float | None) -> str:
    sep = ": " if key in ROW_METRIC_LOG_GROUP else ""
    if value is None:
        return f"{sep}****"
    return f"{sep}{value:.3f}"


def run_eval_layout(
    args: Args,
    dataset: TextDataset,
    tokenizer: GPT2TokenizerWrapper,
    model: GRCEGPT,
    start_pos: int,
    *,
    custom_text: str | None = None,
) -> None:
    """Evaluate the layout on a deterministic slice and print per-row metrics."""

    model_device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    with torch.no_grad():
        (
            context_tokens,
            inputs,
            targets,
            eval_block_length,
            source_label,
            pretty_text,
        ) = _prepare_eval_tokens(
            args,
            dataset,
            tokenizer,
            start_pos=start_pos,
            custom_text=custom_text,
        )

        layout = BatchLayout(args.layout, batch_size=args.batch_size, block_size=eval_block_length)
        _log_layout_warnings(args, layout)

        print(color_text(f"Evaluating layout '{args.layout}' on {source_label}:", Colors.CYAN))
        print(pretty_text)

        xb = inputs.unsqueeze(0).to(model_device)
        yb = targets.unsqueeze(0).to(model_device)
        token_components, position_ids = _embedding_components_with_offsets(
            model,
            xb,
            None,
        )
        future_token_components, _ = _embedding_components_with_offsets(
            model,
            yb,
            None,
        )

        overall_loss_sums = {mode: 0.0 for mode in BATCH_MODES}
        overall_token_counts = {mode: 0 for mode in BATCH_MODES}
        overall_loss_sums["target"] = 0.0
        overall_token_counts["target"] = 0

        def format_metrics(metric_map: dict[str, float | None]) -> str:
            base = _format_eval_metric_value("target", metric_map.get("target"))
            diag = " ".join(
                _format_eval_metric_value(key, metric_map.get(key))
                for key in ROW_METRIC_LOG_KEYS
            )
            return f"{base} {diag}"

        for row_idx, row in enumerate(layout.rows, start=1):
            row_desc = _row_description(layout, row)
            try:
                row_result = _evaluate_row_block(
                    args,
                    model,
                    row,
                    token_components,
                    future_token_components,
                    xb,
                    yb,
                    position_ids,
                    input_tokens=inputs,
                )
            except ValueError as exc:
                print(color_text(f"Row block #{row_idx}: {row_desc} -> error: {exc}", Colors.RED))
                continue

            metrics: dict[str, float | None] = {}
            if row_result.total_tokens > 0:
                metrics["target"] = row_result.total_loss_sum / row_result.total_tokens
            else:
                metrics["target"] = None
            for mode in BATCH_MODES:
                count = row_result.mode_token_counts.get(mode, 0)
                if count > 0:
                    metrics[mode] = row_result.mode_loss_sums.get(mode, 0.0) / count
                else:
                    metrics[mode] = None

            line = format_metrics(metrics)
            print(f"Row block #{row_idx}: {row_desc} (rows={row.rows}) -> {line}")

            weight = max(0, int(row.rows))
            if weight <= 0:
                continue
            overall_loss_sums["target"] += row_result.total_loss_sum * weight
            overall_token_counts["target"] += row_result.total_tokens * weight
            for mode in BATCH_MODES:
                overall_loss_sums[mode] += row_result.mode_loss_sums.get(mode, 0.0) * weight
                overall_token_counts[mode] += row_result.mode_token_counts.get(mode, 0) * weight

        overall_metrics: dict[str, float | None] = {}
        for key in ["target"] + list(BATCH_MODES):
            count = overall_token_counts.get(key, 0)
            if count > 0:
                overall_metrics[key] = overall_loss_sums.get(key, 0.0) / count
            else:
                overall_metrics[key] = None

        overall_line = format_metrics(overall_metrics)
        print(color_text(f"Overall (weighted by rows): {overall_line}", Colors.CYAN))

    if was_training:
        model.train()

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
            if args.command == "create":
                args.model_path_override = checkpoint_path
                args.log_path_override = checkpoint_path.with_suffix(".log")
                return
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_config = payload.get("config")
        if saved_config is None:
            raise ValueError(
                "Checkpoint lacks config metadata; re-save it with the latest format."
            )
        saved = dict(saved_config)
        saved_block_size = int(
            saved.pop("block_size", payload.get("block_size", args.block_size))
        )
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
        args.block_size = saved_block_size
        args.n_layer = config.n_layer
        args.n_head = config.n_head
        args.n_embd = config.n_embd
        args.n_grce = config.n_grce
        args.n_xctx = config.n_xctx
        args.n_rope_axis = getattr(config, "n_rope_axis", args.n_rope_axis)
        args.n_rope_head = getattr(config, "n_rope_head", args.n_rope_head)
        args.vocab_size = config.vocab_size
        args.model_path_override = checkpoint_path
        args.log_path_override = checkpoint_path.with_suffix(".log")
        return

    inferred = ModelGeometry(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_grce=args.n_grce,
        n_xctx=args.n_xctx,
        n_rope_axis=args.n_rope_axis,
        n_rope_head=args.n_rope_head,
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
        self.corpua: list[dict[str, int]] = []
        self.dataset_cache: dict[str, TextDataset] = {}
        self.active_corpus_entry: dict[str, int] | None = None

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

    def _normalize_corpus_entry(self, entry: dict[str, object] | None) -> dict[str, int] | None:
        if not isinstance(entry, dict):
            return None
        name = entry.get("corpus") or entry.get("name")
        if not name:
            return None
        try:
            num_tokens = int(entry.get("num_train_tokens", 0) or 0)
        except (TypeError, ValueError):
            num_tokens = 0
        try:
            max_tokens = int(entry.get("max_train_tokens", num_tokens) or 0)
        except (TypeError, ValueError):
            max_tokens = num_tokens
        try:
            used_tokens = int(entry.get("used_train_tokens", 0) or 0)
        except (TypeError, ValueError):
            used_tokens = 0
        num_tokens = max(0, num_tokens)
        max_tokens = max(num_tokens, max_tokens)
        used_tokens = max(0, min(used_tokens, max_tokens))
        return {
            "corpus": str(name),
            "num_train_tokens": num_tokens,
            "max_train_tokens": max_tokens,
            "used_train_tokens": used_tokens,
        }

    def _convert_legacy_corpora(self, payload: dict[str, object]) -> list[dict[str, int]]:
        datasets = payload.get("datasets")
        if not isinstance(datasets, dict):
            return []
        entries: list[dict[str, int]] = []
        for name, state in datasets.items():
            if not isinstance(state, dict):
                continue
            try:
                train_count = int(state.get("train_count", 0) or 0)
            except (TypeError, ValueError):
                train_count = 0
            try:
                train_cursor = int(state.get("train_cursor", 0) or 0)
            except (TypeError, ValueError):
                train_cursor = 0
            try:
                train_cycles = int(state.get("train_cycles", 0) or 0)
            except (TypeError, ValueError):
                train_cycles = 0
            used_tokens = max(0, train_cycles * max(train_count, 0) + max(0, min(train_cursor, train_count)))
            entries.append(
                {
                    "corpus": str(name),
                    "num_train_tokens": max(0, train_count),
                    "max_train_tokens": max(0, train_count),
                    "used_train_tokens": used_tokens,
                }
            )
        current = payload.get("corpus")
        if current:
            current_name = str(current)
            prioritized = [entry for entry in entries if entry["corpus"] == current_name]
            remaining = [entry for entry in entries if entry["corpus"] != current_name]
            entries = prioritized + remaining
        return entries

    def _load_corpua_from_payload(self, payload: dict | None) -> list[dict[str, int]]:
        if self.corpua:
            return self.corpua
        entries: list[dict[str, int]] = []
        if isinstance(payload, dict):
            raw_list = payload.get("corpua")
            if isinstance(raw_list, list) and raw_list:
                for raw_entry in raw_list:
                    normalized = self._normalize_corpus_entry(raw_entry)
                    if normalized:
                        entries.append(normalized)
            else:
                entries = self._convert_legacy_corpora(payload)
        self.corpua = entries
        return entries

    def _write_corpua_to_payload(self, payload: dict[str, object], corpua: list[dict[str, int]]) -> None:
        payload["corpua"] = [
            {
                "corpus": entry["corpus"],
                "num_train_tokens": int(entry["num_train_tokens"]),
                "max_train_tokens": int(entry["max_train_tokens"]),
                "used_train_tokens": int(entry["used_train_tokens"]),
            }
            for entry in corpua
        ]
        payload.pop("datasets", None)
        payload.pop("corpus", None)

    def _aggregate_corpus_counts(self, name: str, vocab_size: int) -> tuple[int, int]:
        train_cache, test_cache = self._token_cache_paths(name, vocab_size=vocab_size)
        train_tokens = load_cached_tokens("train", train_cache)
        test_tokens = load_cached_tokens("test", test_cache)
        train_count = int(train_tokens.numel())
        test_count = int(test_tokens.numel())
        del train_tokens
        del test_tokens
        return train_count, test_count

    def _select_corpus_entry(self) -> tuple[dict[str, int], bool]:
        if not self.corpua:
            raise RuntimeError(
                "No corpora configured; run 'grce.py corpus --add <name>' before training."
            )
        ready_entry = next(
            (
                entry
                for entry in self.corpua
                if entry.get("used_train_tokens", 0)
                < max(1, int(entry.get("max_train_tokens", entry.get("num_train_tokens", 0)) or 0))
            ),
            None,
        )
        if ready_entry is not None:
            return ready_entry, False
        best_idx = 0
        best_ratio: int | None = None
        for idx, entry in enumerate(self.corpua):
            max_tokens = max(1, int(entry.get("max_train_tokens", entry.get("num_train_tokens", 1)) or 1))
            used_tokens = int(entry.get("used_train_tokens", 0) or 0)
            ratio = used_tokens // max_tokens
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_idx = idx
        return self.corpua[best_idx], True

    def _load_dataset_for_entry(self, entry: dict[str, int]) -> TextDataset:
        name = entry["corpus"]
        cached = self.dataset_cache.get(name)
        if cached is not None:
            return cached
        train_cache, test_cache = self._token_cache_paths(name)
        train_tokens = load_cached_tokens("train", train_cache)
        test_tokens = load_cached_tokens("test", test_cache)
        actual_count = int(train_tokens.numel())
        expected_count = int(entry.get("num_train_tokens", actual_count) or actual_count)
        if expected_count and expected_count != actual_count:
            raise ValueError(
                f"Corpus {name} expected {expected_count:,} train tokens but found {actual_count:,}; rerun 'corpus --add {name}'."
            )
        entry["num_train_tokens"] = actual_count
        entry["max_train_tokens"] = max(int(entry.get("max_train_tokens", actual_count) or actual_count), actual_count)
        dataset = TextDataset(
            train_tokens=train_tokens,
            test_tokens=test_tokens,
            train_text=None,
            test_text=None,
            train_path=train_cache,
            test_path=test_cache,
        )
        self.dataset_cache[name] = dataset
        return dataset

    def _activate_corpus(self) -> TextDataset:
        entry, recycled = self._select_corpus_entry()
        previous = self.active_corpus_entry
        changed = previous is not entry
        self.active_corpus_entry = entry
        dataset = self._load_dataset_for_entry(entry) if changed else self.dataset_cache.get(entry["corpus"])
        if dataset is None:
            dataset = self._load_dataset_for_entry(entry)
        self.dataset = dataset
        self.args.corpus = entry["corpus"]
        if recycled:
            ratio = used_tokens // max(1, max_tokens)
            warning = (
                f"All corpora exhausted; reusing {entry['corpus']} (reuse cycle {ratio + 1})."
            )
            print(color_text(warning, Colors.YELLOW))
        return dataset

    def _ensure_corpus_entry(
        self,
        corpua: list[dict[str, int]],
        name: str,
        *,
        vocab_size: int,
        reuse_multiplier: float,
    ) -> tuple[dict[str, int], bool]:
        reuse_multiplier = max(1.0, float(reuse_multiplier))
        entry = next((item for item in corpua if item["corpus"] == name), None)
        train_count, _ = self._aggregate_corpus_counts(name, vocab_size)
        target_max = max(train_count, int(train_count * reuse_multiplier))
        updated = False
        if entry is None:
            entry = {
                "corpus": name,
                "num_train_tokens": train_count,
                "max_train_tokens": target_max,
                "used_train_tokens": 0,
            }
            corpua.append(entry)
            updated = True
            print(color_text(f"Added corpus {name}", Colors.GREEN))
        else:
            if entry.get("num_train_tokens") != train_count:
                entry["num_train_tokens"] = train_count
                updated = True
            previous_max = entry.get("max_train_tokens", train_count)
            if target_max != previous_max:
                entry["max_train_tokens"] = target_max
                updated = True
        return entry, updated

    def _prepare_corpus(
        self,
        payload: dict | None,
        model_path: pathlib.Path,
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
        corpua = self._load_corpua_from_payload(payload)
        if not corpua:
            raise RuntimeError(
                (
                    f"Checkpoint {model_path} has no corpora registered. "
                    "Run 'grce.py corpus --add <name>' to register datasets before training."
                )
            )
        self.corpua = corpua
        dataset = self._activate_corpus()
        return (
            tokenizer,
            dataset,
            newline_token_id,
            boundary_blocklist,
            default_prompt_boundary,
            tokenizer_json,
        )

    def _token_cache_paths(
        self,
        corpus: str,
        *,
        vocab_size: int | None = None,
    ) -> tuple[pathlib.Path, pathlib.Path]:
        data_dir = pathlib.Path(self.args.data)
        vocab = self.args.vocab_size if vocab_size is None else vocab_size
        train_cache = data_dir / f"{corpus}_tokens_train_{vocab}.pt"
        test_cache = data_dir / f"{corpus}_tokens_test_{vocab}.pt"
        return train_cache, test_cache

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
            atomic_torch_save(payload, target_path)
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

    def cli_json_export(
        self,
        model_path: pathlib.Path,
        payload: dict | None,
    ) -> int:
        if payload is None:
            if not model_path.exists():
                print(color_text(f"Checkpoint {model_path} not found; cannot export JSON.", Colors.RED, bold=True))
                return 1
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            print(color_text("Checkpoint payload is not a metadata dictionary; rerun training with --json.", Colors.RED, bold=True))
            return 1
        json_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"model", "tokenizer_json", "optimizer"}
        }
        target = model_path.with_suffix(".json")
        write_checkpoint_json(json_payload, target)
        print(color_text(f"Wrote checkpoint JSON to {target}", Colors.GREEN))
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
        corpua = self._load_corpua_from_payload(payload)
        names = getattr(self.args, "names", []) or []
        reuse_multiplier = getattr(self.args, "corpus_reuse", 1.0) or 1.0
        add_flag = bool(getattr(self.args, "corpus_add", False))
        set_flag = bool(getattr(self.args, "corpus_set", False))
        updated = False

        if add_flag and names:
            for name in names:
                _, changed = self._ensure_corpus_entry(
                    corpua,
                    name,
                    vocab_size=vocab_size,
                    reuse_multiplier=reuse_multiplier,
                )
                updated = updated or changed

        if set_flag and names:
            for name in names:
                _, changed = self._ensure_corpus_entry(
                    corpua,
                    name,
                    vocab_size=vocab_size,
                    reuse_multiplier=reuse_multiplier,
                )
                updated = updated or changed
            prioritized: list[dict[str, int]] = []
            seen: set[str] = set()
            for name in names:
                for entry in corpua:
                    if entry["corpus"] == name and entry["corpus"] not in seen:
                        prioritized.append(entry)
                        seen.add(entry["corpus"])
                        break
            prioritized.extend(entry for entry in corpua if entry["corpus"] not in seen)
            if prioritized != corpua:
                corpua = prioritized
                updated = True

        self.corpua = corpua
        if updated:
            self._write_corpua_to_payload(payload, corpua)
            atomic_torch_save(payload, model_path)
            print(color_text(f"Saved corpus metadata to {model_path}", Colors.GREEN))

        self._print_corpus_listing(corpua)
        return 0

    def cli_reset(
        self,
        model_path: pathlib.Path,
        payload: dict | None,
    ) -> int:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Checkpoint {model_path} not found; cannot reset counters."
            )
        if payload is None:
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint payload must be a dictionary")
        baseline_loss_history = payload.get("loss_history")
        baseline_cycles = int(payload.get("completed_cycles", 0) or 0)
        baseline_steps = int(payload.get("total_steps", 0) or 0)
        payload["loss_history"] = []
        payload["completed_cycles"] = 0
        payload["total_steps"] = 0
        payload["train_wall_seconds"] = 0.0
        atomic_torch_save(payload, model_path)
        print(
            color_text(
                (
                    f"Reset checkpoint {model_path.name}: cycles {baseline_cycles}→0,"
                    f" steps {baseline_steps}→0, cleared {len(baseline_loss_history or [])} history entries"
                ),
                Colors.GREEN,
            )
        )
        return 0

    def _print_corpus_listing(self, corpua: Sequence[dict[str, int]]) -> None:
        if not corpua:
            print(color_text("No corpora registered in checkpoint", Colors.YELLOW))
            return
        print(color_text("Registered corpora:", Colors.CYAN))
        for idx, entry in enumerate(corpua, start=1):
            name = entry.get("corpus", "<unknown>")
            num_tokens = int(entry.get("num_train_tokens", 0) or 0)
            max_tokens = max(num_tokens, int(entry.get("max_train_tokens", 0) or 0))
            used_tokens = int(entry.get("used_train_tokens", 0) or 0)
            reuse = (max_tokens / num_tokens) if num_tokens > 0 else 0.0
            progress = (used_tokens / max_tokens) if max_tokens > 0 else 0.0
            line = (
                f"{idx:>2}. {name}: used {used_tokens:,} / {max_tokens:,}"
                f" ({progress:.1%}) of {num_tokens:,} tokens"
            )
            if reuse > 1.0:
                line += f" [reuse x{reuse:.2f}]"
            print(line)

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
            print(color_text(f"Logfile: {log_path}", Colors.CYAN))
            dataset_commands = {"train", "try", "report", "test", "eval", "profile", "prompts"}
            requires_checkpoint = self.args.command in {"train", "try", "report", "test", "eval", "profile", "prompts", "reset", "corpus"}
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
            if requires_checkpoint and self.args.command != "create" and not model_path.exists():
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
            if self.args.command == "reset":
                return self.cli_reset(model_path, payload)
            if self.args.command == "json":
                return self.cli_json_export(model_path, payload if isinstance(payload, dict) else None)

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
                (
                    tokenizer,
                    dataset,
                    newline_token_id,
                    boundary_blocklist,
                    default_prompt_boundary,
                    tokenizer_json,
                ) = self._prepare_corpus(payload, model_path)
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
                color_text(
                    f"Trainable model params: {total_params:,}; "
                    f"excl. embeddings: {non_emb_params:,}",
                    Colors.BLUE,
                )
            )
            tok_vecs = config.vocab_size
            emb_vectors = tok_vecs
            emb_params = embedding_params
            print(
                color_text(
                    f"Learned embedding vectors: {emb_vectors} "
                    f"(token={tok_vecs}); params={emb_params:,}",
                    Colors.BLUE,
                )
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
                prompt_text_for_tokens = apply_prompt_prefix(
                    self.args.prompt,
                    enabled=getattr(self.args, "prompt_prefix", True),
                )
                prompt_tokens = tokenizer.encode(prompt_text_for_tokens)
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
            total_train_tokens = 0
            payload = getattr(self.args, "checkpoint_payload_override", None)
            optimizer_state = None
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
                        self.args.completed_cycles = int(payload.get("completed_cycles", 0) or 0)
                        upgraded = upgrade_state_dict(payload["model"])
                        payload["model"] = upgraded
                        model.load_state_dict(upgraded)
                        total_steps = int(payload.get("total_steps", 0))
                        loss_history = list(payload.get("loss_history", []))
                        total_train_wall = float(payload.get("train_wall_seconds", 0.0))
                        total_train_tokens = int(payload.get("total_train_tokens", 0) or 0)
                        prompt_registry = PromptRegistry(
                            tokenizer,
                            payload.get("prompts"),
                        )
                        if self.args.checkpoint_optimizer:
                            optimizer_state = payload.get("optimizer")
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
                total_train_tokens = int(meta.get("total_train_tokens", 0) or 0)
                loss_history = []
                write_timer = Timer().start()
                atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        "total_steps": total_steps,
                        "loss_history": loss_history,
                        "config": asdict(args_to_model_geometry(self.args)),
                        "block_size": self.args.block_size,
                        "train_wall_seconds": total_train_wall,
                        "total_train_tokens": total_train_tokens,
                        "prompts": prompt_registry.serialize() if prompt_registry else None,
                        "tokenizer_json": tokenizer_json,
                        "completed_cycles": int(meta.get("completed_cycles", 0)),
                        "corpua": self.corpua,
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

            if getattr(self.args, "_cycles_is_delta", False):
                delta = int(getattr(self.args, "_cycles_delta", 0))
                base_cycles = int(getattr(self.args, "completed_cycles", 0))
                self.args.cycles = base_cycles + delta

            if self.args.command == "create":
                checkpoint_payload = {
                    "model": model.state_dict(),
                    "total_steps": 0,
                    "loss_history": [],
                    "config": asdict(args_to_model_geometry(self.args)),
                    "block_size": self.args.block_size,
                    "train_wall_seconds": 0.0,
                    "total_train_tokens": 0,
                    "prompts": prompt_registry.serialize(),
                    "tokenizer_json": tokenizer_json,
                    "completed_cycles": 0,
                    "corpua": self.corpua,
                }
                atomic_torch_save(checkpoint_payload, model_path)
                print(
                    color_text(
                        f"Created new checkpoint at {model_path}; run 'corpus --add <name>' before training.",
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
                custom_text = None
                raw_text = getattr(self.args, "text", None)
                if raw_text:
                    joined = " ".join(raw_text).strip()
                    if joined:
                        custom_text = joined
                run_test_slice(
                    args=self.args,
                    dataset=dataset,
                    tokenizer=tokenizer,
                    model=model,
                    start_pos=self.args.test_start,
                    custom_text=custom_text,
                )
                return

            if self.args.command == "eval":
                custom_text = None
                raw_text = getattr(self.args, "text", None)
                if raw_text:
                    joined = " ".join(raw_text).strip()
                    if joined:
                        custom_text = joined
                if custom_text:
                    run_eval_layout(
                        args=self.args,
                        dataset=dataset,
                        tokenizer=tokenizer,
                        model=model,
                        start_pos=self.args.eval_start,
                        custom_text=custom_text,
                    )
                    return
                rand_runs = max(0, int(getattr(self.args, "eval_random", 0)))
                if rand_runs > 0:
                    total = len(dataset.test_tokens)
                    if total <= 0:
                        raise ValueError("Test corpus is empty; cannot run random evaluations")
                    rng = random.Random()
                    window = max(1, total - (self.args.block_size + 1))
                    for run_idx in range(rand_runs):
                        start_pos = rng.randint(0, window - 1)
                        print(color_text(f"[eval random #{run_idx + 1}] offset {start_pos}", Colors.BLUE))
                        run_eval_layout(
                            args=self.args,
                            dataset=dataset,
                            tokenizer=tokenizer,
                            model=model,
                            start_pos=start_pos,
                            custom_text=None,
                        )
                    return
                run_eval_layout(
                    args=self.args,
                    dataset=dataset,
                    tokenizer=tokenizer,
                    model=model,
                    start_pos=self.args.eval_start,
                    custom_text=None,
                )
                return

            if self.args.command == "profile":
                optimizer = torch.optim.AdamW(
                    _optimizer_param_groups(model, self.args.weight_decay),
                    lr=self.args.lr_base,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_eps,
                )
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
                    batch_size=self.args.batch_size,
                    device=device,
                )
                return

            def build_optimizer() -> torch.optim.Optimizer:
                return torch.optim.AdamW(
                    _optimizer_param_groups(model, self.args.weight_decay),
                    lr=self.args.lr_base,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_eps,
                )

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
            completed_cycles = getattr(self.args, "completed_cycles", 0)
            cycle_start = completed_cycles + 1
            cycle_end = max(completed_cycles, self.args.cycles)
            for cycle in range(cycle_start, cycle_end + 1):
                dataset = self._activate_corpus()
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
                        block_size=self.args.block_size,
                    )
                    for _ in range(self.args.steps)
                ]
                print()
                pod_path = pathlib.Path("/.podname")
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
                corpus_entry = self.active_corpus_entry or {}
                used_tokens = int(corpus_entry.get("used_train_tokens", 0) or 0)
                num_tokens = int(corpus_entry.get("num_train_tokens", dataset.train_tokens.numel()) or dataset.train_tokens.numel())
                max_tokens = int(corpus_entry.get("max_train_tokens", num_tokens) or num_tokens)
                used_pct = (used_tokens / max(1, num_tokens)) * 100.0 if num_tokens > 0 else 0.0
                max_pct = (max_tokens / max(1, num_tokens)) * 100.0 if num_tokens > 0 else 0.0
                corpus_status = (
                    f"Active corpus: {self.args.corpus} ({used_tokens:,} / {num_tokens:,}"
                    f" = {used_pct:.2f}% tokens used; max = {max_pct:.2f}%)"
                )
                print(color_text(corpus_status, Colors.CYAN))
                per_run_idx = cycle
                print(
                    color_text(
                        f"[{label}] Training Cycle {per_run_idx}/{self.args.cycles}. "
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
                    tokens_consumed,
                ) = train_model(
                    self.args,
                    model,
                    dataset,
                    device,
                    self.args.steps,
                    self.args.batch_size,
                    self.args.eval_interval,
                    total_steps,
                    optimizer,
                    prompt_tokens,
                    self.args.generate,
                    total_train_tokens,
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
                entry = self.active_corpus_entry
                if entry is not None:
                    entry["used_train_tokens"] = int(entry.get("used_train_tokens", 0)) + int(tokens_consumed)
                pure_train = Timer().add(train_timer).sub(eval_timer)
                acc_train.add(pure_train)
                acc_eval.add(eval_timer)
                total_train_wall += train_timer.wall_secs
                total_train_tokens += int(tokens_consumed)
                update_wall: float | None = None
                if not self.args.skip_model_update:
                    include_optimizer = (
                        self.args.checkpoint_optimizer
                        and not self.args.restart_optimizer
                        and cycle < cycle_end
                    )
                    update_timer = Timer().start()
                    checkpoint_payload: dict[str, Any] = {
                        "model": model.state_dict(),
                        "corpua": self.corpua,
                        "total_steps": total_steps,
                        "loss_history": loss_history,
                        "config": asdict(args_to_model_geometry(self.args)),
                        "block_size": self.args.block_size,
                        "train_wall_seconds": total_train_wall,
                        "total_train_tokens": total_train_tokens,
                        "prompts": prompt_registry.serialize() if prompt_registry else None,
                        "tokenizer_json": tokenizer_json,
                        "completed_cycles": cycle,
                        **(
                            {"optimizer": optimizer.state_dict()}
                            if include_optimizer
                            else {}
                        ),
                    }
                    atomic_torch_save(checkpoint_payload, model_path)
                    if getattr(self.args, "train_json", False):
                        json_payload = {
                            key: value
                            for key, value in checkpoint_payload.items()
                            if key not in {"model", "tokenizer_json", "optimizer"}
                        }
                        write_checkpoint_json(json_payload, model_path.with_suffix(".json"))
                    update_wall = update_timer.stop().wall_secs
                cycle_part = color_text(f"[cycle {cycle} (wall/cpu/gpu)]", Colors.CYAN)
                train_part = color_text(f" train: {pure_train};", Colors.MAGENTA)
                eval_part = color_text(f" eval: {eval_timer};", Colors.GREEN)
                if self.args.skip_model_update:
                    updated_part = color_text(" model update skipped; flushing logs.", Colors.YELLOW)
                else:
                    wall_text = f"{update_wall:.2f}s" if update_wall is not None else "?"
                    updated_part = color_text(
                        f" model update: {wall_text}; flushing logs.",
                        Colors.YELLOW,
                    )
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
                optimizer = None
            self.args.completed_cycles = cycle_end

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
