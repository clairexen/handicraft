"""Mini-language parser for GRCE batch layouts.

This module hosts :class:`BatchLayout`, a helper that interprets the batch
layout strings described in the project notes.  The class handles the
preprocessor (``(...)`` alternatives), row/column range specs, ``*`` expansion
markers, and exposes the resolved concrete layout plus serialization helpers.

The implementation mirrors the specification in the user instructions:

* ``A+B+...`` concatenates row groups; each group is ``ROWS[SEGMENTS]``.
* ``SEGMENTS`` are slash-separated ``COLS``+``MODE`` tokens (``16e/32d``).
* Ranges use ``START-END`` (inclusive) and may be prefixed by ``*`` to allow
  dynamic expansion when additional rows/columns are needed.
* Bare ``*`` behaves like ``*0-0`` for the baseline but participates in
  expansion with a constant weight of 1.
* Parentheses with ``|`` act as a textual pre-processor: ``A(B|C)D`` randomly
  expands to either ``ABD`` or ``ACD`` before the parser runs.

The resolved layout is stored as structured data and can be re-serialized into
the concrete (fully deterministic) layout string.
"""

from __future__ import annotations

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

    @classmethod
    def parse(cls, token: str) -> "CountSpec":
        token = token.strip()
        if not token:
            raise LayoutParseError("Missing size spec")
        expandable = token.startswith("*")
        constant_weight = False
        if expandable:
            token = token[1:]
            if not token:
                constant_weight = True
        if not token:
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
        if minimum < 0 or maximum < 0:
            raise LayoutParseError("Negative sizes are not supported")
        if maximum < minimum:
            raise LayoutParseError(f"Invalid range {minimum}-{maximum}")
        return cls(minimum, maximum, expandable, constant_weight)

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


def _split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, ch in enumerate(text):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth < 0:
                raise LayoutParseError("Unbalanced ']' in layout string")
        elif ch == sep and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    if depth != 0:
        raise LayoutParseError("Unbalanced '[' in layout string")
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

    def expand_weight(self) -> int:
        if not self.spec.expandable:
            return 0
        if self.spec.constant_weight:
            return 1
        return max(1, self.value)

    def expand(self) -> None:
        if not self.spec.expandable:
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
        candidates = [item for item in items if item.spec.expandable]
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
        terms = _split_top_level(processed.replace(" ", ""), "+")
        if not terms:
            raise LayoutParseError("Layout string is empty")
        self.row_specs = [_parse_row_spec(term) for term in terms]
        self.rows = self._materialize_rows()

    def _materialize_rows(self) -> list[RowLayout]:
        row_allocs: list[_CountAllocation] = []
        for spec in self.row_specs:
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
        for spec, allocation in zip(self.row_specs, row_allocs):
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

        parts: list[str] = []
        for row in self.rows:
            segment_bits = []
            for segment in row.segments:
                letter = _MODE_LETTERS.get(segment.mode, segment.mode[0])
                segment_bits.append(f"{segment.columns}{letter}")
            parts.append(f"{row.rows}[{'/'.join(segment_bits)}]")
        return "+".join(parts)

    def expanded_rows(self) -> list[list[SegmentLayout]]:
        """Return the per-row segments with rows fully expanded."""

        rows: list[list[SegmentLayout]] = []
        for group in self.rows:
            for _ in range(group.rows):
                rows.append([SegmentLayout(seg.mode, seg.columns) for seg in group.segments])
        return rows
