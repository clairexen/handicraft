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

    vocab_size: int = 30000 # GPT-2 base supports ~50k merges.
    n_pos: int = 1024       # GPT-2 base uses 1024 tokens.
    n_layer: int = 12       # GPT-2 base uses 12 layers.
    n_head: int = 12        # GPT-2 base uses 12 attention heads.
    n_width: int = 768       # GPT-2 base uses 768 embedding dims.
    n_rope: int = 0         # Number of Q/K dims using RoPE (0 => full head width).
    n_grce: int = 64        # Narrow GRCE context dims.
    n_xctx: int = 1536      # Wide XCTX context dims.
    n_query: int = 1        # Number of query vectors per head.
    use_gmlp: bool = False
    use_rope_xl: bool = False
    use_rope_vr: bool = False
    use_rope_vr_all: bool = False
    use_sane: bool = False

    @property
    def block_size(self) -> int:
        return self.n_pos


MODEL_GEOMETRY_DEFAULTS = ModelGeometry()


@dataclass
class Defaults:
    """Default Settings (override with CLI args)"""

    vocab_size: int = MODEL_GEOMETRY_DEFAULTS.vocab_size
    n_pos: int = MODEL_GEOMETRY_DEFAULTS.n_pos
    block_size: int = MODEL_GEOMETRY_DEFAULTS.n_pos
    n_layer: int = MODEL_GEOMETRY_DEFAULTS.n_layer
    n_head: int = MODEL_GEOMETRY_DEFAULTS.n_head
    n_width: int = MODEL_GEOMETRY_DEFAULTS.n_width
    n_rope: int = MODEL_GEOMETRY_DEFAULTS.n_rope
    n_grce: int = MODEL_GEOMETRY_DEFAULTS.n_grce
    n_xctx: int = MODEL_GEOMETRY_DEFAULTS.n_xctx
    n_query: int = MODEL_GEOMETRY_DEFAULTS.n_query
    use_gmlp: bool = MODEL_GEOMETRY_DEFAULTS.use_gmlp
    use_rope_xl: bool = MODEL_GEOMETRY_DEFAULTS.use_rope_xl
    use_rope_vr: bool = MODEL_GEOMETRY_DEFAULTS.use_rope_vr
    use_rope_vr_all: bool = MODEL_GEOMETRY_DEFAULTS.use_rope_vr_all
    use_sane: bool = MODEL_GEOMETRY_DEFAULTS.use_sane
    corpus: str | None = None
    steps: int = 100
    cycles: int = 100
    batch_size: int = 256
    layout: str = "*[*1e=*1d=*1f=*2t2x=*1n=*1r]"
    eval_interval: int = 10
    dropout: float = 0.05
    detach_span: int = 0
    detach_think_span: int = 0
    rng_seed: int = 1234
    rng_cycle_only: bool = False
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
    generate_with_decode: bool = False
    align_articles: bool = False
    allow_oversize: bool = False

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

global_runtime_args: Args | None = None


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
    think_factor: int = 1
    sane_x: int = 1
    sane_x_last_only: bool = False
    layer_repeat: int = 1
    layer_top_only: bool = False
    think_last_only: bool = False
    metric_mode: str | None = None
    hide_typed_metrics: bool = False
    extra_metrics: tuple[str, ...] = ()
    suppress_default_metric: bool = False
    loss_input_stream: bool = False
    loss_output_stream: bool = False
    drop_count: int = 0
    disable_sane: bool = False
    sane_z: int = 1
    sane_z_strict: bool = False


@dataclass(frozen=True)
class RowModifiers:
    detach_kv_cache: bool = False
    detach_span: int | None = None
    no_detach_ctx: bool = False
    train_transformer_only: bool = False
    train_recurrent_only: bool = False
    halt_rope: bool = False
    disable_sane: bool = False
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
    think_factor: int = 1
    think_last_only: bool = False
    layer_repeat: int = 1
    layer_top_only: bool = False
    metric_mode: str | None = None
    hide_typed_metrics: bool = False
    extra_metrics: tuple[str, ...] = ()
    suppress_default_metric: bool = False
    loss_input_stream: bool = False
    loss_output_stream: bool = False
    drop_count: int = 0
    disable_sane: bool = False
    token_columns_override: int | None = None
    sane_z: int = 1
    sane_x: int = 1
    sane_x_last_only: bool = False
    sane_z_strict: bool = False

    def token_columns(self) -> int:
        if self.token_columns_override is not None:
            return self.token_columns_override
        if self.think_factor <= 1:
            return self.columns
        return (self.columns // self.think_factor)


def _metric_template_has_coords(template: str) -> bool:
    return "$" in template


def _format_metric_template(
    template: str,
    x_value: int | None,
    z_value: int | None,
) -> str:
    replacements = [x_value, z_value]
    result = template
    for value in replacements:
        idx = result.find("$")
        if idx == -1:
            break
        replacement = str(int(value)) if value is not None else "0"
        result = result[:idx] + replacement + result[idx + 1 :]
    return result


def _segment_metric_coordinate_names(segment: SegmentLayout, template: str) -> list[str]:
    x_count = max(1, int(getattr(segment, "sane_x", 1) or 1))
    z_count = max(1, int(getattr(segment, "sane_z", 1) or 1))
    names: list[str] = []
    for x in range(x_count):
        for z in range(z_count):
            names.append(_format_metric_template(template, x, z))
    return names


@dataclass
class BlockLayout:
    rows: int
    segments: list[SegmentLayout]
    modifiers: RowModifiers | None = None

    def total_columns(self) -> int:
        return sum(segment.columns for segment in self.segments)

    def token_span(self) -> int:
        columns = sum(segment.token_columns() for segment in self.segments)
        if columns <= 0 or self.rows <= 0:
            return 0
        return self.rows * (columns + 1)

    def total_positions(self) -> int:
        return sum(segment.token_columns() for segment in self.segments)


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
    depth = 0
    for index, ch in enumerate(body):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth < 0:
                raise LayoutParseError("Unbalanced ']' in segment string")
        elif ch == ",":
            continue
        if depth > 0:
            continue
        if ch in "=#":
            token = body[start:index].strip()
            if token:
                parts.append((token, connector))
            connector = ch
            start = index + 1
    if depth != 0:
        raise LayoutParseError("Unbalanced '[' in segment string")
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
    mode_with_metrics = text[index:]
    if not mode_with_metrics:
        raise LayoutParseError("Missing mode in segment")
    metric_text = ""
    gt_index = mode_with_metrics.find('>')
    if gt_index != -1:
        metric_text = mode_with_metrics[gt_index:]
        mode_token = mode_with_metrics[:gt_index]
    else:
        mode_token = mode_with_metrics
    if not mode_token:
        raise LayoutParseError("Missing mode in segment")
    base_mode_char = mode_token[0].lower()
    think_factor = 1
    think_last_only = False
    sane_x = 1
    sane_x_last_only = False
    sane_z = 1
    sane_z_strict = False
    layer_repeat = 1
    layer_top_only = False
    suffix_pattern = re.compile(r"(\d+)([xXyYzZ])$")
    while True:
        match = suffix_pattern.search(mode_token)
        if not match:
            break
        count = int(match.group(1))
        marker = match.group(2)
        mode_token = mode_token[: match.start()]
        if marker in {"x", "X"}:
            if base_mode_char in {"f", "t"}:
                if count not in (2, 3, 4):
                    raise LayoutParseError("Think modifiers only support 2x/3x/4x")
                if think_factor != 1:
                    raise LayoutParseError("Think modifier specified multiple times")
                think_factor = count
                think_last_only = marker.isupper()
            elif base_mode_char == "d":
                if count not in (2, 3, 4):
                    raise LayoutParseError("Think modifiers only support 2x/3x/4x")
                if sane_x != 1:
                    raise LayoutParseError("Think modifier specified multiple times")
                sane_x = count
                sane_x_last_only = marker.isupper()
            else:
                raise LayoutParseError(
                    "Think modifiers are only supported for forward/think or decode segments"
                )
            continue
        if marker in {"z", "Z"}:
            if base_mode_char != "d":
                raise LayoutParseError("Z modifiers are only supported for decode segments")
            if count <= 0:
                raise LayoutParseError("Z modifiers require a positive integer")
            if sane_z != 1:
                raise LayoutParseError("Z modifier specified multiple times")
            sane_z = count
            sane_z_strict = marker.isupper()
            continue
        if marker in {"y", "Y"}:
            if count <= 0:
                raise LayoutParseError("Layer repeat modifiers require a positive integer")
            if layer_repeat != 1:
                raise LayoutParseError("Layer repeat modifier specified multiple times")
            layer_repeat = count
            layer_top_only = marker.isupper()
            continue
        raise LayoutParseError(f"Unsupported suffix modifier '{marker}' in segment '{text}'")
    hide_typed_metrics = False
    bias_input_loss = False
    bias_output_loss = False
    drop_count = 0
    disable_sane = False
    while mode_token:
        tail = mode_token[-1]
        if tail == "h":
            raise LayoutParseError("Lowercase 'h' is reserved as a row modifier; use 'H' to hide metrics")
        if tail == "H":
            hide_typed_metrics = True
            mode_token = mode_token[:-1]
            continue
        if tail in {"b", "B"}:
            if tail == "b":
                bias_input_loss = True
            else:
                bias_output_loss = True
            mode_token = mode_token[:-1]
            continue
        if tail == "P":
            if drop_count:
                raise LayoutParseError("Drop modifier specified multiple times")
            idx_end = len(mode_token) - 1
            idx_start = idx_end
            while idx_start > 0 and mode_token[idx_start - 1].isdigit():
                idx_start -= 1
            count_text = mode_token[idx_start:idx_end]
            drop_count = int(count_text) if count_text else 1
            if drop_count <= 0:
                raise LayoutParseError("Drop modifier requires a positive count")
            mode_token = mode_token[:idx_start]
            continue
        if tail == "S":
            disable_sane = True
            mode_token = mode_token[:-1]
            continue
        break
    if hide_typed_metrics and not mode_token:
        raise LayoutParseError("Hide-metric modifier requires a base mode")
    if not mode_token:
        raise LayoutParseError("Missing mode in segment")
    mode_char = mode_token[0]
    context_enabled = mode_char.islower()
    metric_mode: str | None = None
    mode_key = mode_char.lower()
    if mode_key == "t":
        if not context_enabled:
            raise LayoutParseError("Think segments must keep context enabled (use lowercase 't')")
        metric_mode = "think"
        mode_key = "f"
    if mode_key not in _MODE_ALIASES:
        raise LayoutParseError(f"Unsupported mode '{mode_token}'")
    if drop_count and mode_key != "e":
        raise LayoutParseError("Drop modifier 'P' is only supported for encode segments")
    if disable_sane and mode_key not in {"e", "d", "r"}:
        raise LayoutParseError("'S' modifier only valid for encode/decode/reverse segments")
    if think_factor > 1:
        if mode_key != "f":
            raise LayoutParseError("Think multipliers are only valid for forward segments")
        if not context_enabled:
            raise LayoutParseError("Think multipliers require context-enabled segments")
    size_spec = CountSpec.parse(size_token or "1")
    if connector not in {None, "=", "#"}:
        raise LayoutParseError(f"Unsupported segment connector '{connector}'")
    extra_metrics: list[str] = []
    suppress_default = False
    if metric_text:
        if metric_text.startswith('>>'):
            suppress_default = True
            metric_body = metric_text[2:]
        elif metric_text.startswith('>'):
            metric_body = metric_text[1:]
        else:
            raise LayoutParseError("Metric annotations must start with '>' or '>>'")
        if not metric_body:
            raise LayoutParseError("Metric annotations require at least one name")
        extra_metrics = [part for part in metric_body.split('>') if part]
        if not extra_metrics:
            raise LayoutParseError("Metric annotations require non-empty names")
    return SegmentSpec(
        size_spec,
        _MODE_ALIASES[mode_key],
        context_enabled,
        connector,
        think_factor=think_factor,
        sane_x=sane_x,
        sane_x_last_only=sane_x_last_only,
        think_last_only=think_last_only,
        layer_repeat=layer_repeat,
        layer_top_only=layer_top_only,
        metric_mode=metric_mode,
        hide_typed_metrics=hide_typed_metrics,
        extra_metrics=tuple(extra_metrics),
        suppress_default_metric=suppress_default,
        loss_input_stream=bias_input_loss,
        loss_output_stream=bias_output_loss,
        drop_count=drop_count,
        disable_sane=disable_sane,
        sane_z=sane_z,
        sane_z_strict=sane_z_strict,
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
    halt_rope = False
    disable_sane = False
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
        if ch == "h":
            halt_rope = True
            raw_parts.append("h")
            idx += 1
            continue
        if ch == "S":
            disable_sane = True
            raw_parts.append("S")
            idx += 1
            continue
        raise LayoutParseError(f"Unknown row modifier '{ch}' in '{text}'")
    raw = "".join(raw_parts)
    return RowModifiers(
        detach_kv_cache=detach_kv,
        detach_span=detach_span,
        no_detach_ctx=no_detach_ctx,
        train_transformer_only=train_transformer_only,
        train_recurrent_only=train_recurrent_only,
        halt_rope=halt_rope,
        disable_sane=disable_sane,
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
        extra_names: set[str] = set()
        for row in self.rows:
            for segment in row.segments:
                templates = getattr(segment, "extra_metrics", ())
                for template in templates:
                    if _metric_template_has_coords(template):
                        extra_names.update(
                            _segment_metric_coordinate_names(segment, template)
                        )
                    else:
                        extra_names.add(template)
        self.extra_metric_names = sorted(extra_names)

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
        segments: list[SegmentLayout] = []
        for spec, allocation in zip(specs, allocations):
            columns = allocation.value
            think_factor = max(1, int(getattr(spec, "think_factor", 1) or 1))
            if think_factor > 1:
                columns = (columns // think_factor) * think_factor
            sane_z = max(1, int(getattr(spec, "sane_z", 1) or 1))
            sane_x = max(1, int(getattr(spec, "sane_x", 1) or 1))
            sane_x_last_only = bool(getattr(spec, "sane_x_last_only", False))
            sane_z_strict = bool(getattr(spec, "sane_z_strict", False))
            token_columns_override = None
            if getattr(spec, "mode", "") == "decode" and sane_z > 1:
                base_columns = columns
                token_columns_override = base_columns
                if base_columns <= 0:
                    columns = 0
                else:
                    columns = (base_columns - 1) * sane_z + 1
            segments.append(
                SegmentLayout(
                    spec.mode,
                    columns,
                    spec.context_enabled,
                    spec.connector,
                    think_factor=think_factor,
                    think_last_only=bool(getattr(spec, "think_last_only", False)),
                    layer_repeat=max(1, int(getattr(spec, "layer_repeat", 1) or 1)),
                    layer_top_only=bool(getattr(spec, "layer_top_only", False)),
                    metric_mode=getattr(spec, "metric_mode", None),
                    hide_typed_metrics=getattr(spec, "hide_typed_metrics", False),
                    extra_metrics=getattr(spec, "extra_metrics", ()),
                    suppress_default_metric=getattr(spec, "suppress_default_metric", False),
                    loss_input_stream=getattr(spec, "loss_input_stream", False),
                    loss_output_stream=getattr(spec, "loss_output_stream", False),
                    drop_count=max(0, int(getattr(spec, "drop_count", 0) or 0)),
                    disable_sane=bool(getattr(spec, "disable_sane", False)),
                    token_columns_override=token_columns_override,
                    sane_z=sane_z,
                    sane_x=sane_x,
                    sane_x_last_only=sane_x_last_only,
                    sane_z_strict=sane_z_strict,
                )
            )
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
                if segment.metric_mode == "think":
                    letter = "t" if segment.context_enabled else "T"
                else:
                    letter = _MODE_LETTERS.get(segment.mode, segment.mode[0])
                    if not segment.context_enabled:
                        letter = letter.upper()
                hide_suffix = ""
                if getattr(segment, "hide_typed_metrics", False):
                    hide_suffix = "H"
                think_suffix = ""
                if getattr(segment, "think_factor", 1) and segment.think_factor > 1:
                    suffix_letter = "X" if getattr(segment, "think_last_only", False) else "x"
                    think_suffix = f"{segment.think_factor}{suffix_letter}"
                decode_think_suffix = ""
                if segment.mode == "decode" and getattr(segment, "sane_x", 1) > 1:
                    suffix_letter = "X" if getattr(segment, "sane_x_last_only", False) else "x"
                    decode_think_suffix = f"{segment.sane_x}{suffix_letter}"
                layer_suffix = ""
                if getattr(segment, "layer_repeat", 1) and segment.layer_repeat > 1:
                    suffix_letter = "Y" if getattr(segment, "layer_top_only", False) else "y"
                    layer_suffix = f"{segment.layer_repeat}{suffix_letter}"
                bias_suffix = ""
                if getattr(segment, "loss_output_stream", False):
                    bias_suffix += "B"
                if getattr(segment, "loss_input_stream", False):
                    bias_suffix += "b"
                drop_suffix = ""
                drop_count = getattr(segment, "drop_count", 0)
                if drop_count:
                    prefix = f"{drop_count}" if drop_count > 1 else ""
                    drop_suffix = f"{prefix}P"
                depth_suffix = ""
                if getattr(segment, "sane_z", 1) and segment.sane_z > 1:
                    suffix_letter = "Z" if getattr(segment, "sane_z_strict", False) else "z"
                    depth_suffix = f"{segment.sane_z}{suffix_letter}"
                sane_suffix = "S" if getattr(segment, "disable_sane", False) else ""
                metric_suffix = ""
                extra = getattr(segment, "extra_metrics", ())
                if extra:
                    prefix = ">>" if getattr(segment, "suppress_default_metric", False) else ">"
                    metric_suffix = prefix + ">".join(extra)
                bit = f"{segment.columns}{letter}{hide_suffix}{layer_suffix}{think_suffix}{decode_think_suffix}{bias_suffix}{drop_suffix}{depth_suffix}{sane_suffix}{metric_suffix}"
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
                        SegmentLayout(
                            seg.mode,
                            seg.columns,
                            seg.context_enabled,
                            seg.connector,
                            think_factor=seg.think_factor,
                            think_last_only=seg.think_last_only,
                            layer_repeat=seg.layer_repeat,
                            layer_top_only=seg.layer_top_only,
                            metric_mode=seg.metric_mode,
                            hide_typed_metrics=seg.hide_typed_metrics,
                            extra_metrics=seg.extra_metrics,
                            suppress_default_metric=seg.suppress_default_metric,
                            loss_input_stream=seg.loss_input_stream,
                            loss_output_stream=seg.loss_output_stream,
                            drop_count=seg.drop_count,
                            disable_sane=seg.disable_sane,
                        )
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
from typing import Any, Dict, List, Tuple, Sequence, Callable, Mapping, NamedTuple


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
    def flag_present(*flags: str) -> bool:
        for flag in flags:
            if any(arg == flag or arg.startswith(f"{flag}=") for arg in raw_cli_args):
                return True
        return False

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
        "--n-pos",
        type=int,
        default=DEFAULTS.n_pos,
        help="Maximum supported sequence length (context window)",
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
        "--n-query",
        type=int,
        default=DEFAULTS.n_query,
        help="Number of independent query projections per attention head.",
    )
    model_group.add_argument(
        "--n-width",
        type=int,
        default=DEFAULTS.n_width,
        dest="n_width",
        help="Transformer width (GPT-2 base uses 768); must be a multiple of n_head.",
    )
    model_group.add_argument(
        "--n-rope",
        type=int,
        default=DEFAULTS.n_rope,
        help="Number of Q/K features using Rotary Position Embedding (0 => full head width; must be even).",
    )
    model_group.add_argument(
        "--use-gmlp",
        action="store_true",
        default=DEFAULTS.use_gmlp,
        help="Use gated MLP feed-forward blocks (adds a second 4x projection as a multiplicative gate)",
    )
    model_group.add_argument(
        "--use-rope-xl",
        action="store_true",
        default=DEFAULTS.use_rope_xl,
        help=(
            "Enable the RoPE-XL mapping that compresses relative offsets beyond N/2 into a capped range; "
            "requires even --n-pos"
        ),
    )
    model_group.add_argument(
        "--use-rope-vr",
        action="store_true",
        default=DEFAULTS.use_rope_vr,
        help=(
            "Rotate value vectors for half the attention heads so relative position information can propagate"
        ),
    )
    model_group.add_argument(
        "--use-rope-vr-all",
        action="store_true",
        default=DEFAULTS.use_rope_vr_all,
        help="Rotate value vectors for every attention head instead of only half",
    )
    model_group.add_argument(
        "--use-sane",
        action="store_true",
        default=DEFAULTS.use_sane,
        help=(
            "Enable the Self-And-Next Encoder (SANE) grid hooks that propagate residuals between positions"
        ),
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
        "--small",
        action="store_true",
        help=(
            "Shortcut for --n-layer 6 --n-head 4 --n-width 256 --n-grce 48 --n-xctx 720"
        ),
    )
    model_group.add_argument(
        "--tiny",
        action="store_true",
        help=(
            "Shortcut for --vocab-size 600 --batch-size 12 --n-pos 10 --n-layer 3 --n-head 2 "
            "--n-width 8 --n-grce 4 --n-xctx 9 --steps 2 --eval-interval 1"
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
        "--block-size",
        type=int,
        default=None,
        help="Actual tokens-per-sample used during train/eval (defaults to --n-pos)",
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
        "--detach-think-span",
        type=int,
        default=DEFAULTS.detach_think_span,
        help=(
            "When think spans are present, isolate groups of N spans by detaching the recurrent"
            " state and KV cache replicas at their boundaries (0 disables detaching)."
        ),
    )
    training_group.add_argument(
        "--no-detach-ctx",
        action="store_true",
        help="Keep gradients through the recurrent GRCE context even when spans trigger",
    )
    training_group.add_argument(
        "--allow-oversize",
        action="store_true",
        default=DEFAULTS.allow_oversize,
        help="Permit --block-size to exceed --n-pos when probing generalization",
    )
    training_group.add_argument(
        "--align-articles",
        action="store_true",
        default=DEFAULTS.align_articles,
        help=(
            "When sampling random corpus windows, align them to the <|----|> article separator token"
            " and resample if multiple separators remain"
        ),
    )
    training_group.add_argument(
        "--rng-seed",
        type=int,
        default=DEFAULTS.rng_seed,
        help="Base RNG seed (0 disables deterministic seeding)",
    )
    training_group.add_argument(
        "--rng-cycle-only",
        action="store_true",
        default=DEFAULTS.rng_cycle_only,
        help="Derive RNG seeds only from --rng-seed and the cycle index (ignore consumed tokens)",
    )
    parser.add_argument(
        "--allow-shape-mismatch-load",
        action="store_true",
        help=(
            "When loading checkpoints, reuse overlapping parameter slices even if shapes differ."
            " By default shape mismatches cause an error."
        ),
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
        "--generate-with-decode",
        action="store_true",
        default=DEFAULTS.generate_with_decode,
        help="Use decode-mode attention when generating samples",
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
        "--log-all",
        action="store_true",
        help="Force log/ANSI files to update even for read-only commands (eval, size, etc.)",
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
    eval_parser.add_argument(
        "--verbose",
        dest="eval_verbose",
        action="store_true",
        help="Print the evaluated slice and per-row metrics",
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
    import_parser = subparsers.add_parser(
        "import",
        help="Convert an existing checkpoint to the configured geometry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    import_parser.set_defaults(command="import")
    import_parser.add_argument(
        "import_source",
        type=pathlib.Path,
        help="Path to the checkpoint .pt file to import",
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
            "\nPlease specify a command (train, try, report, test, eval, profile, size, corpus, create, import, or prompts).\n",
        )


    # --------------------------------------------------------
    # Normalize, tweak, and check global options

    if args.small:
        if not flag_present("--n-layer"):
            args.n_layer = 6
        if not flag_present("--n-head"):
            args.n_head = 4
        if not flag_present("--n-width"):
            args.n_width = 256
        if not flag_present("--n-grce"):
            args.n_grce = 48
        if not flag_present("--n-xctx"):
            args.n_xctx = 720

    if args.tiny:
        if not flag_present("--vocab-size"):
            args.vocab_size = 600
        if not flag_present("--batch-size"):
            args.batch_size = 12
        if not flag_present("--n-pos"):
            args.n_pos = 10
        if not flag_present("--n-layer"):
            args.n_layer = 3
        if not flag_present("--n-head"):
            args.n_head = 2
        if not flag_present("--n-width"):
            args.n_width = 8
        if not flag_present("--n-grce"):
            args.n_grce = 4
        if not flag_present("--n-xctx"):
            args.n_xctx = 9
        if not flag_present("--steps"):
            args.steps = 2
        if not flag_present("--eval-interval"):
            args.eval_interval = 1

    if args.n_pos <= 0:
        parser.error("--n-pos must be positive")
    args._block_size_defined = args.block_size is not None
    if args.block_size is None:
        args.block_size = args.n_pos
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.n_rope < 0:
        parser.error("--n-rope must be non-negative")
    if args.n_rope and args.n_rope % 2 != 0:
        parser.error("--n-rope must be even")
    if args.n_width % args.n_head != 0:
        parser.error("--n-width must be divisible by --n-head")
    head_dim = args.n_width // args.n_head
    if args.n_rope == 0 and head_dim % 2 != 0:
        parser.error(
            "Default RoPE span requires an even per-head width; either adjust n_width/n_head or set --n-rope"
        )
    if args.n_rope and args.n_rope > head_dim:
        parser.error("--n-rope must be <= per-head width (n_width / n_head)")
    if args.block_size > args.n_pos:
        if not getattr(args, "allow_oversize", False):
            parser.error("--block-size must be <= --n-pos (pass --allow-oversize to override)")
        else:
            print(
                "Warning: allowing block_size to exceed n_pos; attention masks remain limited by n_pos"
            )
    if args.log_row_details:
        args.log_step_details = True

    args.prompt = normalize_prompt(args.prompt)
    args.checkpoint_dirty = False

    use_rope_xl = bool(getattr(args, "use_rope_xl", False))
    use_rope_vr = bool(getattr(args, "use_rope_vr", False))
    use_rope_vr_all = bool(getattr(args, "use_rope_vr_all", False))
    if use_rope_xl and (args.n_pos % 2 != 0):
        raise ValueError("--use-rope-xl requires an even --n-pos")
    if use_rope_vr and use_rope_vr_all:
        raise ValueError("--use-rope-vr and --use-rope-vr-all are mutually exclusive")
    if use_rope_vr:
        if args.n_head % 2 != 0:
            raise ValueError("--use-rope-vr requires an even --n-head")
    if getattr(args, "use_sane", False) and (args.n_width % 2 != 0):
        raise ValueError("--use-sane requires an even --n-width")
    args.checkpoint_dirty = False

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

    return args


def args_to_model_geometry(args: Args):
    """Project the parsed CLI namespace into :class:`ModelGeometry` metadata.

    ``Runtime`` uses this helper when saving checkpoints or describing models
    so downstream tools and :func:`grce_main` can reload consistent geometry.
    """

    return ModelGeometry(
        vocab_size=args.vocab_size,
        n_pos=args.n_pos,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_width=args.n_width,
        n_rope=args.n_rope,
        n_grce=args.n_grce,
        n_xctx=args.n_xctx,
        n_query=getattr(args, "n_query", MODEL_GEOMETRY_DEFAULTS.n_query),
        use_gmlp=getattr(args, "use_gmlp", MODEL_GEOMETRY_DEFAULTS.use_gmlp),
        use_rope_xl=getattr(args, "use_rope_xl", MODEL_GEOMETRY_DEFAULTS.use_rope_xl),
        use_rope_vr=getattr(args, "use_rope_vr", MODEL_GEOMETRY_DEFAULTS.use_rope_vr),
        use_rope_vr_all=getattr(args, "use_rope_vr_all", MODEL_GEOMETRY_DEFAULTS.use_rope_vr_all),
        use_sane=getattr(args, "use_sane", MODEL_GEOMETRY_DEFAULTS.use_sane),
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
    upgraded: dict[str, torch.Tensor]
    if not needs_upgrade:
        upgraded = dict(state)
    else:
        upgraded = {}
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
    return _upgrade_control_embedding_rows(upgraded)


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
        config.n_width // 2,
        config.n_xctx // 2,
        max(config.n_width // 4, config.n_xctx // config.n_layer),
    )


def _build_geometry(config: GeometryLike, n_pos: int) -> list[tuple[str, str, int]]:
    """Assemble the (label, description, value) tuples for ``size`` reports."""

    return [
        ("V", "vocab size", config.vocab_size),
        ("B", "block size", n_pos),
        ("L", "transform layers", config.n_layer),
        ("H", "attention heads", config.n_head),
        ("Q", "queries per head", getattr(config, "n_query", 1)),
        ("E", "embedding width", config.n_width),
        ("G", "grce width", config.n_grce),
        ("X", "xctx width", config.n_xctx),
        ("U", "inner xctx width", _get_inner_xctx_width(config)),
    ]

def _expected_sections(config: GeometryLike, n_pos: int) -> list[tuple[str, str, list[dict]]]:
    """Return analytic section breakdown consumed by :func:`grce_cmd_size`."""

    V = config.vocab_size
    B = n_pos
    L = config.n_layer
    H = config.n_head
    E = config.n_width
    G = config.n_grce
    X = config.n_xctx
    Q = getattr(config, "n_query", 1)
    sections: list[tuple[str, str, list[dict]]] = []

    def eval_items(items):
        for item in items:
            item["count"] = eval(
                item["formula"],
                {
                    "V": config.vocab_size,
                    "B": n_pos,
                    "L": config.n_layer,
                    "H": config.n_head,
                    "E": config.n_width,
                    "G": config.n_grce,
                    "X": config.n_xctx,
                    "Q": Q,
                },
            )
        return items

    global_items = eval_items([
        {"label": "token embeddings", "formula": "V * E"},
        {"label": "special embeddings", "formula": "0 * E"},
        {"label": "grce embeddings", "formula": "0 * G"},
    ])
    sections.append(("embeddings", "Embeddings", global_items))

    transformer_defs = [
        {
            "label": "attn qkv",
            "formula": "L * ((2+Q) * (E*E + E))",
        },
        {
            "label": "attn proj",
            "formula": "L * (E*(E*Q) + E)",
        },
        {
            "label": "ffn fc1",
            "formula": "L * (4*E*E + 4*E)",
        },
        {
            "label": "ffn fc2",
            "formula": "L * (4*E*E + E)",
        },
    ]
    if getattr(config, "use_gmlp", False):
        transformer_defs.append(
            {
                "label": "ffn gate",
                "formula": "L * (4*E*E + 4*E)",
            }
        )
    transformer_items = eval_items(transformer_defs)
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
    ffn_gate = 0
    for block in model.core.blocks:
        attn_qkv += sum(
            _module_param_count(getattr(block.attn, attr))
            for attr in ("key", "query", "value")
        )
        attn_proj += _module_param_count(block.attn.proj)
        ffn_fc1 += _module_param_count(block.ff.fc1)
        ffn_fc2 += _module_param_count(block.ff.fc2)
        if getattr(block.ff, "fc_gate", None) is not None:
            ffn_gate += _module_param_count(block.ff.fc_gate)
    counts[("transformer", "attn qkv")] = attn_qkv
    counts[("transformer", "attn proj")] = attn_proj
    counts[("transformer", "ffn fc1")] = ffn_fc1
    counts[("transformer", "ffn fc2")] = ffn_fc2
    if ffn_gate:
        counts[("transformer", "ffn gate")] = ffn_gate

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
    E = config.n_width
    G = config.n_grce
    X = config.n_xctx
    U = _get_inner_xctx_width(config)
    Q = getattr(config, "n_query", 1)
    extra = 4 if getattr(config, "use_gmlp", False) else 0
    transformer_factor = 11 + Q + extra
    estimates = [
        ("transformer", transformer_factor * L * E * E, f"({transformer_factor}*L*E^2)"),
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


def _resolved_rope_width(config: GeometryLike) -> int:
    """Return the effective RoPE width per head after defaults are applied."""

    head_dim = max(1, config.n_width // max(1, config.n_head))
    rope_dim = int(config.n_rope)
    if rope_dim <= 0:
        rope_dim = head_dim
    return rope_dim


def _standard_rope_rho(n_pos: int, rope_dim: int) -> float | None:
    """Compute the classic RoPE ρ value for ``n_pos`` tokens and ``rope_dim`` dims."""

    if n_pos <= 0 or rope_dim <= 0:
        return None
    return float(n_pos) ** (2.0 / float(rope_dim))


def _print_rope_reports(config: GeometryLike, n_pos: int) -> None:
    """Emit RoPE ρ diagnostics for the size report."""

    rope_dim = _resolved_rope_width(config)
    standard_rho = _standard_rope_rho(n_pos, rope_dim)
    print(color_text("Positional Encoding", Colors.CYAN, bold=True))
    print(f"  resolved n_rope dims : {rope_dim}")
    if standard_rho is None:
        print("  roh (standard RoPE): n/a")
    else:
        print(f"  roh (standard RoPE, N={n_pos}): {standard_rho:.6f}")


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
    geometry = _build_geometry(args, args.n_pos)
    sections = _append_summary_section(_expected_sections(args, args.n_pos))
    _print_geometry(geometry)
    print()
    _print_rope_reports(args, args.n_pos)
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

if os.environ.get("GRCE_DETECT_ANOMALY") == "1":
    torch.autograd.set_detect_anomaly(True)
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


class RMSNormNoAffine(nn.Module):
    """Parameter-free RMSNorm for SANE loop iterations."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(rms + self.eps)
        return x * scale


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


class ThinkEmbeddingLibrary(nn.Module):
    """Holds learnable embeddings for explicit think-step annotations."""

    def __init__(self, dim: int, think_spans: Sequence[int] | None = None) -> None:
        super().__init__()
        spans = tuple(sorted({span for span in (think_spans or (2, 3, 4)) if span >= 2}))
        if not spans:
            raise ValueError("ThinkEmbeddingLibrary requires at least one think span >= 2")
        self.spans = spans
        entries: dict[str, nn.Parameter] = {}
        for total in self.spans:
            for index in range(1, total + 1):
                key = self._key(index, total)
                entries[key] = nn.Parameter(torch.zeros(dim))
        self.embeddings = nn.ParameterDict(entries)
        self.more_embedding = nn.Parameter(torch.zeros(dim))
        self.last_embedding = nn.Parameter(torch.zeros(dim))

    @staticmethod
    def _key(index: int, total: int) -> str:
        return f"{index}/{total}"

    def get(self, index: int, total: int) -> torch.Tensor:
        if total not in self.spans:
            raise ValueError(f"ThinkEmbeddingLibrary does not support think{total}x mode")
        if index < 1 or index > total:
            raise ValueError(f"Think embedding index {index} out of range for think{total}x")
        key = self._key(index, total)
        try:
            return self.embeddings[key]
        except KeyError as exc:
            raise ValueError(f"Missing think embedding for slot {key}") from exc

    def status(self, is_last: bool) -> torch.Tensor:
        return self.last_embedding if is_last else self.more_embedding

    def sequence(self, total: int) -> torch.Tensor:
        if total not in self.spans:
            raise ValueError(f"ThinkEmbeddingLibrary does not support think{total}x mode")
        slots: list[torch.Tensor] = []
        for index in range(1, total + 1):
            slots.append(self.get(index, total) + self.status(index == total))
        return torch.stack(slots, dim=0)


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


ARTICLE_ALIGN_RETRY_LIMIT = 16


@dataclass
class TextDataset:
    """Stores train/test tensors and samples random spans from each corpus."""

    train_tokens: torch.Tensor
    test_tokens: torch.Tensor
    train_text: str | None
    test_text: str | None
    train_path: pathlib.Path
    test_path: pathlib.Path
    article_separator_token_id: int | None = None
    align_articles: bool = False

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
        future_margin: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, int]], torch.Tensor]:
        if columns <= 0 or rows <= 0:
            raise ValueError("Row sampling requires positive columns and rows")
        if future_margin < 0:
            raise ValueError("future_margin must be non-negative")
        tokens = self._tokens_for_split(split)
        total = int(tokens.numel())
        seq_span = columns + 1 + future_margin
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
            chunk, chunk_start = self._aligned_window(
                tokens,
                start,
                seq_span,
                rng=rng,
                allow_resample=True,
            )
            windows.append(chunk)
            end = chunk_start + seq_span - 1
            metadata.append(
                {
                    "token_start": chunk_start,
                    "token_end": end % total,
                    "token_span": seq_span,
                    "wrapped": 1 if end >= total else 0,
                    "total_tokens": total,
                    "row": row_idx + 1,
                }
            )
        stacked = torch.stack(windows)
        x = stacked[:, :columns].contiguous().to(device)
        y = stacked[:, 1 : columns + 1].contiguous().to(device)
        future = torch.empty(rows, 0, dtype=x.dtype, device=device)
        if future_margin > 0:
            future = stacked[:, columns + 1 :].contiguous().to(device)
        return x, y, metadata, future

    def _aligned_window(
        self,
        tokens: torch.Tensor,
        start: int,
        needed: int,
        *,
        rng: random.Random | None,
        allow_resample: bool,
    ) -> tuple[torch.Tensor, int]:
        total = int(tokens.numel())
        if total <= 0:
            raise ValueError("No tokens available for alignment")
        align_token = self.article_separator_token_id if self.align_articles else None
        current = start % total
        attempts = 0
        chunk, _ = self._slice_with_wrap(tokens, None, current, needed)
        if align_token is None:
            return chunk, current
        while True:
            matches = (chunk == align_token).nonzero(as_tuple=False).flatten()
            if matches.numel() == 0:
                return chunk, current
            shift = int(matches[0].item())
            adjusted = (current + shift) % total
            chunk, _ = self._slice_with_wrap(tokens, None, adjusted, needed)
            matches = (chunk == align_token).nonzero(as_tuple=False).flatten()
            if matches.numel() <= 1 or not allow_resample:
                return chunk, adjusted
            attempts += 1
            if attempts >= ARTICLE_ALIGN_RETRY_LIMIT:
                return chunk, adjusted
            rng_obj = rng or random
            current = rng_obj.randint(0, total - 1)
            chunk, _ = self._slice_with_wrap(tokens, None, current, needed)

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

CONTROL_NONE = 0
CONTROL_PREDICT_NEXT = 1
CONTROL_PREDICT_PREV = 2
CONTROL_FIND_SELF = 3
CONTROL_EMBEDDING_ROWS = 4


def _expand_tensor_to_parity(
    tensor: torch.Tensor, full_dim: int, parity: int
) -> torch.Tensor:
    """Scatter last-dim features into even/odd slots of a wider tensor."""

    if parity not in (0, 1):
        raise ValueError("parity must be 0 (even) or 1 (odd)")
    shape = tensor.shape[:-1] + (full_dim,)
    expanded = tensor.new_zeros(shape)
    expanded[..., parity::2] = tensor
    return expanded


def _select_parity_features(tensor: torch.Tensor, parity: int) -> torch.Tensor:
    """Return a contiguous view of the even or odd slots of ``tensor``."""

    if parity not in (0, 1):
        raise ValueError("parity must be 0 (even) or 1 (odd)")
    return tensor[..., parity::2].contiguous()


def _partial_state_dict_load(
    model: nn.Module, source_state: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    """Copy overlapping tensor slices from ``source_state`` into ``model`` state."""

    target_state = model.state_dict()
    total = len(target_state)
    reused = 0
    missing = 0
    resized = 0
    incompatible = 0
    missing_examples: list[str] = []
    resized_examples: list[str] = []
    incompatible_examples: list[str] = []
    for name, target_value in list(target_state.items()):
        source_value = source_state.get(name)
        if source_value is None:
            missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(name)
            continue
        if source_value.dim() != target_value.dim():
            incompatible += 1
            if len(incompatible_examples) < 5:
                incompatible_examples.append(
                    f"{name}: checkpoint {tuple(source_value.shape)} vs model {tuple(target_value.shape)}"
                )
            continue
        overlap = tuple(min(s, t) for s, t in zip(source_value.shape, target_value.shape))
        if any(length <= 0 for length in overlap):
            incompatible += 1
            if len(incompatible_examples) < 5:
                incompatible_examples.append(
                    f"{name}: checkpoint {tuple(source_value.shape)} vs model {tuple(target_value.shape)}"
                )
            continue
        target_copy = target_value.clone()
        slices = tuple(slice(0, length) for length in overlap)
        converted = source_value.to(dtype=target_value.dtype)
        target_copy[slices] = converted[slices]
        target_state[name] = target_copy
        reused += 1
        if source_value.shape != target_value.shape:
            resized += 1
            if len(resized_examples) < 5:
                resized_examples.append(
                    f"{name}: checkpoint {tuple(source_value.shape)} -> model {tuple(target_value.shape)}"
                )
    unused = 0
    if hasattr(source_state, "keys"):
        try:
            source_keys = set(source_state.keys())
            target_keys = set(target_state.keys())
            unused = len(source_keys - target_keys)
        except TypeError:
            unused = 0
    model.load_state_dict(target_state)
    return {
        "total": total,
        "reused": reused,
        "missing": missing,
        "resized": resized,
        "incompatible": incompatible,
        "unused": unused,
        "missing_examples": missing_examples,
        "resized_examples": resized_examples,
        "incompatible_examples": incompatible_examples,
}


def _strip_state_entries(
    state: Mapping[str, torch.Tensor],
    *,
    predicate: Callable[[str], bool],
) -> Mapping[str, torch.Tensor]:
    if not hasattr(state, "items"):
        return state
    filtered = [(key, value) for key, value in state.items() if not predicate(key)]
    if len(filtered) == len(state):
        return state
    state_type = type(state)
    try:
        return state_type(filtered)
    except Exception:  # pragma: no cover - fallback for exotic OrderedDicts
        return dict(filtered)


def _strip_sane_parameters(state: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    return _strip_state_entries(state, predicate=lambda key: ".sane_" in key)


def _upgrade_control_embedding_rows(
    state: dict[str, torch.Tensor],
    *,
    expected_rows: int = CONTROL_EMBEDDING_ROWS,
) -> dict[str, torch.Tensor]:
    if expected_rows <= 0:
        return state
    upgraded = dict(state)
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if not key.endswith("control_emb.weight"):
            continue
        rows, dim = value.shape
        if rows >= expected_rows:
            continue
        pad = value.new_zeros(expected_rows - rows, dim)
        upgraded[key] = torch.cat([value, pad], dim=0)
    return upgraded


def _control_embedding_rows(state: Mapping[str, torch.Tensor]) -> int | None:
    if not hasattr(state, "get"):
        return None
    weight = state.get("core.control_emb.weight")
    if isinstance(weight, torch.Tensor):
        return weight.shape[0]
    return None


def _load_checkpoint_state(
    model: nn.Module, state: Mapping[str, torch.Tensor], *, allow_partial: bool
) -> dict[str, object]:
    """Attempt to load ``state`` strictly; fall back to partial loading on mismatch."""

    total = len(model.state_dict())
    try:
        filtered_state = {
            key: value
            for key, value in state.items()
            if not key.endswith("attn.tril")
        }
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        if unexpected_keys:
            print(color_text(f"Warning: unexpected keys during load: {unexpected_keys}", Colors.YELLOW))
        return {
            "success": True,
            "partial": False,
            "total": total,
            "reused": total - len(missing_keys),
            "missing": len(missing_keys),
            "resized": 0,
            "incompatible": 0,
            "unused": len(unexpected_keys),
            "error": None,
        }
    except RuntimeError as err:
        if not allow_partial:
            raise RuntimeError(
                "Checkpoint load failed due to parameter shape mismatch. "
                "Delete the checkpoint or rerun with --allow-shape-mismatch-load to continue with partial weights."
            ) from err
        summary = _partial_state_dict_load(model, state)
        summary.update({
            "success": False,
            "partial": True,
            "error": err,
        })
        return summary


def _log_partial_checkpoint_warning(summary: Mapping[str, object]) -> None:
    """Explain how many tensors were reused vs. reinitialized."""

    reused = int(summary.get("reused", 0))
    total = int(summary.get("total", 0))
    missing = int(summary.get("missing", 0))
    resized = int(summary.get("resized", 0))
    incompatible = int(summary.get("incompatible", 0))
    unused = int(summary.get("unused", 0))
    parts = [f"reused {reused}/{total} tensors"]
    if missing:
        parts.append(f"initialized {missing} new tensors")
    if resized:
        parts.append(f"cropped/extended {resized} tensors to fit new shapes")
    if incompatible:
        parts.append(f"skipped {incompatible} incompatible tensors")
    if unused:
        parts.append(f"ignored {unused} checkpoint-only tensors")
    main = ", ".join(parts)
    print(
        color_text(
            f"Checkpoint partially loaded ({main}). New parameters keep their default initialization.",
            Colors.YELLOW,
            bold=True,
        )
    )
    resized_examples = summary.get("resized_examples", []) or []
    if resized_examples:
        print(color_text("Examples of tensors that were cropped/extended:", Colors.YELLOW))
        for entry in resized_examples:
            print(f"  - {entry}")
    incompatible_examples = summary.get("incompatible_examples", []) or []
    if incompatible_examples:
        print(color_text("Examples of tensors that could not be mapped:", Colors.YELLOW))
        for entry in incompatible_examples:
            print(f"  - {entry}")
    missing_examples = summary.get("missing_examples", []) or []
    if missing_examples:
        print(color_text("Examples of tensors only present in the new model:", Colors.YELLOW))
        for name in missing_examples:
            print(f"  - {name}")
    print(color_text("Optimizer state discarded due to architecture changes.", Colors.YELLOW))
    print(
        color_text(
            "Set --allow-shape-mismatch-load to enable this behavior explicitly in future runs.",
            Colors.YELLOW,
        )
    )


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
        if config.n_width % config.n_head != 0:
            raise ValueError("--n-width must be divisible by --n-head")
        self.n_head = config.n_head
        self.n_query = max(1, int(getattr(config, "n_query", 1) or 1))
        self.total_heads = self.n_head * self.n_query
        self.head_dim = config.n_width // config.n_head
        raw_rope = max(0, int(getattr(config, "n_rope", 0) or 0))
        self.raw_n_rope = raw_rope
        resolved_rope = self._resolve_rope_dim(raw_rope)
        self.total_rope_dim = resolved_rope
        self.rope_dim = resolved_rope
        self.use_rope_xl = bool(getattr(config, "use_rope_xl", False))
        self.use_rope_vr = bool(getattr(config, "use_rope_vr", False))
        self.use_rope_vr_all = bool(getattr(config, "use_rope_vr_all", False))
        if self.use_rope_vr and (self.n_head % 2 != 0):
            raise ValueError("RoPE-VR requires an even --n-head value")
        if self.use_rope_vr and self.use_rope_vr_all:
            raise ValueError("use_rope_vr and use_rope_vr_all cannot both be enabled")
        self.rope_xl_half_window = float(config.n_pos) / 2.0 if self.use_rope_xl else 0.0
        self.key = nn.Linear(config.n_width, config.n_width)
        self.query = nn.Linear(config.n_width, config.n_width * self.n_query)
        self.value = nn.Linear(config.n_width, config.n_width)
        proj_in = config.n_width * self.n_query
        self.proj = nn.Linear(proj_in, config.n_width)
        self.dropout = nn.Dropout(config.dropout)
        if self.use_rope_vr_all:
            self.vr_head_count = self.n_head
        elif self.use_rope_vr:
            self.vr_head_count = self.n_head // 2
        else:
            self.vr_head_count = 0
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.n_pos, config.n_pos))
        )
        if self.rope_dim:
            if self.rope_dim > self.head_dim:
                raise ValueError("Resolved RoPE width cannot exceed per-head width")
            base = max(1, config.n_pos)
            idx = torch.arange(0, self.rope_dim, 2, dtype=torch.float32)
            inv_freq = torch.pow(torch.tensor(float(base), dtype=torch.float32), -idx / self.rope_dim)
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
            self.register_buffer("rope_cos_cached", torch.empty(0), persistent=False)
            self.register_buffer("rope_sin_cached", torch.empty(0), persistent=False)
            self._build_rope_cache(base)
        else:
            self.register_buffer("rope_inv_freq", torch.empty(0), persistent=False)
            self.register_buffer("rope_cos_cached", torch.empty(0), persistent=False)
            self.register_buffer("rope_sin_cached", torch.empty(0), persistent=False)

    def _resolve_rope_dim(self, raw: int) -> int:
        head_dim = self.head_dim
        if raw > 0:
            if raw % 2 != 0:
                raise ValueError("--n-rope must be even")
            return raw
        if head_dim % 2 != 0:
            raise ValueError("Per-head width must be even when using default RoPE span")
        return max(0, head_dim)

    def _ensure_tril_capacity(self, size: int, device: torch.device) -> torch.Tensor:
        if size <= 0:
            return self.tril
        tril = self.tril
        current = tril.size(0)
        target = int(size)
        if current >= target and tril.device == device:
            return tril
        if current >= target and tril.device != device:
            tril = tril.to(device)
            self.tril = tril
            return tril
        new_size = max(target, current * 2 if current > 0 else target)
        dtype = tril.dtype
        tril = torch.tril(torch.ones(new_size, new_size, dtype=dtype, device=device))
        self.tril = tril
        return tril

    def _build_rope_cache(self, max_seq: int) -> None:
        if self.rope_dim <= 0:
            return
        device = self.rope_inv_freq.device
        pos = torch.arange(max_seq, dtype=torch.float32, device=device)
        freqs = torch.outer(pos, self.rope_inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.rope_cos_cached = cos.cpu()
        self.rope_sin_cached = sin.cpu()

    def _ensure_rope_cache(self, needed: int) -> None:
        if self.rope_dim <= 0:
            return
        cached = self.rope_cos_cached.size(0)
        if needed <= cached:
            return
        self._build_rope_cache(needed)

    def _rope_cos_sin_positions(
        self,
        positions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope_dim <= 0:
            raise RuntimeError("RoPE cache requested but rope_dim is zero")
        if positions.numel() == 0:
            empty = torch.empty(0, self.rope_dim // 2, device=device, dtype=dtype)
            return empty, empty
        if self.use_rope_xl:
            mapped = self._rope_xl_map_positions(
                positions.to(device=device, dtype=torch.float32)
            )
            inv = self.rope_inv_freq.to(device=device, dtype=torch.float32)
            freqs = torch.outer(mapped, inv)
            cos = torch.cos(freqs).to(dtype=dtype)
            sin = torch.sin(freqs).to(dtype=dtype)
            return cos, sin
        max_pos = int(positions.max().item()) + 1
        self._ensure_rope_cache(max_pos)
        index = positions.to(device=self.rope_cos_cached.device, dtype=torch.long)
        cos = self.rope_cos_cached.index_select(0, index).to(device=device, dtype=dtype)
        sin = self.rope_sin_cached.index_select(0, index).to(device=device, dtype=dtype)
        return cos, sin

    def _rope_xl_map_positions(self, positions: torch.Tensor) -> torch.Tensor:
        half = self.rope_xl_half_window
        if half <= 0:
            return positions
        abs_vals = positions.abs()
        mask = abs_vals <= half
        denom = abs_vals.clamp_min(1e-9)
        half_tensor = positions.new_full((), half)
        ratio = half_tensor / denom
        scaled = (3.0 * half_tensor) - (2.0 * half_tensor * torch.sqrt(ratio))
        mapped = torch.where(mask, abs_vals, scaled)
        return mapped * positions.sign()

    def _rope_cos_sin_with_offsets(
        self,
        positions: torch.Tensor,
        position_offsets: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_offsets is None:
            return self._rope_cos_sin_positions(positions, device, dtype)
        offsets = position_offsets.to(device=device, dtype=torch.long)
        if offsets.dim() != 1 or offsets.size(0) != batch_size:
            raise ValueError("position_offsets must be 1D with batch_size entries")
        base = positions.to(device=device, dtype=torch.long)
        grid = base.unsqueeze(0) + offsets.view(-1, 1)
        flat = grid.reshape(-1)
        cos, sin = self._rope_cos_sin_positions(flat, device, dtype)
        cos = cos.view(batch_size, base.size(0), -1)
        sin = sin.view(batch_size, base.size(0), -1)
        return cos, sin

    def _apply_rope(self, tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        rope_dim = self.rope_dim
        if rope_dim <= 0:
            return tensor
        orig_shape = tensor.shape
        rope_slice = tensor[..., :rope_dim]
        other = tensor[..., rope_dim:]
        rope_view = rope_slice.view(*orig_shape[:-1], rope_dim // 2, 2)
        x_even = rope_view[..., 0]
        x_odd = rope_view[..., 1]
        if cos.dim() == 2:
            cos = cos.view(1, cos.size(0), 1, cos.size(1))
            sin = sin.view(1, sin.size(0), 1, sin.size(1))
        elif cos.dim() == 3:
            cos = cos.view(cos.size(0), cos.size(1), 1, cos.size(2))
            sin = sin.view(sin.size(0), sin.size(1), 1, sin.size(2))
        else:
            raise ValueError("RoPE cos/sin tensors must be rank 2 or 3")
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1).reshape(*orig_shape[:-1], rope_dim)
        return torch.cat((rotated, other), dim=-1)

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
        rope_positions: torch.Tensor | None = None,
        position_offsets: torch.Tensor | None = None,
        sane_group_ids: torch.Tensor | None = None,
        sane_z_indices: torch.Tensor | None = None,
        sane_active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        head_dim = self.head_dim
        key_states = self.key(x).view(B, T, self.n_head, head_dim)
        value_states = self.value(x).view(B, T, self.n_head, head_dim)
        query_states = self.query(x).view(B, T, self.n_head, self.n_query, head_dim)
        query_states = query_states.reshape(B, T, self.total_heads, head_dim)
        if rope_positions is None:
            base_positions = torch.arange(T, device=x.device, dtype=torch.long)
        else:
            base_positions = rope_positions.to(device=x.device, dtype=torch.long)
            if base_positions.dim() != 1 or base_positions.size(0) != T:
                raise ValueError("rope_positions must match sequence length")
        if self.rope_dim:
            cos, sin = self._rope_cos_sin_with_offsets(
                base_positions,
                position_offsets,
                x.device,
                query_states.dtype,
                x.size(0),
            )
            query_states = self._apply_rope(query_states, cos, sin)
            key_states = self._apply_rope(key_states, cos, sin)
            if self.vr_head_count > 0:
                vr_slice = value_states[:, :, : self.vr_head_count, :]
                rotated_values = self._apply_rope(vr_slice, cos, sin)
                value_states = torch.cat(
                    [rotated_values, value_states[:, :, self.vr_head_count :, :]], dim=2
                )
        if self.n_query > 1:
            key_attn = key_states.unsqueeze(3).expand(-1, -1, -1, self.n_query, -1)
            value_attn = value_states.unsqueeze(3).expand(-1, -1, -1, self.n_query, -1)
            key_attn = key_attn.reshape(B, T, self.total_heads, head_dim)
            value_attn = value_attn.reshape(B, T, self.total_heads, head_dim)
        else:
            key_attn = key_states.reshape(B, T, self.total_heads, head_dim)
            value_attn = value_states.reshape(B, T, self.total_heads, head_dim)
        q = query_states.transpose(1, 2)
        k_local = key_attn.transpose(1, 2)
        v_local = value_attn.transpose(1, 2)

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
            if self.n_query > 1:
                cat_keys = cat_keys.repeat_interleave(self.n_query, dim=1)
                cat_values = cat_values.repeat_interleave(self.n_query, dim=1)
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
        base_allowed: torch.Tensor | None = None
        if attn_mode not in {"encode", "decode", "reverse", "noattn"}:
            raise ValueError(f"Unknown attention mode: {attn_mode}")
        if attn_mode == "decode" and not full_attention:
            tril = self._ensure_tril_capacity(T, x.device)
            block_mask = tril[:T, :T] == 0
            base_allowed = tril[:T, :T] == 1
        elif attn_mode == "reverse" and not full_attention:
            tril = self._ensure_tril_capacity(T, x.device)
            block_mask = tril[:T, :T].transpose(0, 1) == 0
        elif attn_mode == "noattn":
            eye = torch.eye(T, dtype=torch.bool, device=x.device)
            block_mask = ~eye
        if (
            block_mask is not None
            and attn_mode == "decode"
            and sane_group_ids is not None
            and sane_z_indices is not None
            and cache_len == 0
            and sane_group_ids.numel() == T
            and sane_z_indices.numel() == T
        ):
            groups = sane_group_ids.to(x.device, dtype=torch.long)
            z_idx = sane_z_indices.to(x.device, dtype=torch.long)
            allowed = torch.zeros(T, T, dtype=torch.bool, device=x.device)
            col_index = torch.arange(T, device=x.device)
            for j in range(T):
                same_group = groups == groups[j]
                earlier_zero = (groups < groups[j]) & (z_idx == 0)
                cond = same_group | earlier_zero
                allowed[j, cond] = True
            if sane_active_mask is not None:
                active_vec = sane_active_mask.to(x.device, dtype=torch.bool)
                if active_vec.dim() != 1 or active_vec.size(0) != T:
                    raise ValueError("sane_active_mask must match sequence length")
                row_mask = active_vec.view(-1, 1)
                col_mask = active_vec.view(1, -1)
                both_active = row_mask & col_mask
                if base_allowed is None:
                    tril = self._ensure_tril_capacity(T, x.device)
                    base_allowed = tril[:T, :T] == 1
                allowed = torch.where(both_active, allowed, base_allowed)
            block_mask = ~allowed
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
            key_states,
            value_states,
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

        attn_output = attn_output.transpose(1, 2).contiguous()
        y = attn_output.view(B, T, self.total_heads * head_dim)
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
        position_index: int | None = None,
    ) -> tuple[torch.Tensor, "LayerCache"]:
        if x.size(1) != 1:
            raise ValueError("Incremental attention expects a single-token sequence")
        B, T, C = x.shape
        head_dim = self.head_dim
        key_states = self.key(x).view(B, T, self.n_head, head_dim)
        value_states = self.value(x).view(B, T, self.n_head, head_dim)
        query_states = self.query(x).view(B, T, self.n_head, self.n_query, head_dim)
        query_states = query_states.reshape(B, T, self.total_heads, head_dim)
        if self.rope_dim:
            pos_value = cache.length if position_index is None else position_index
            positions = torch.tensor([pos_value], device=x.device, dtype=torch.long)
            cos, sin = self._rope_cos_sin_positions(
                positions,
                x.device,
                query_states.dtype,
            )
            query_states = self._apply_rope(query_states, cos, sin)
            key_states = self._apply_rope(key_states, cos, sin)
        if self.n_query > 1:
            key_attn = key_states.unsqueeze(3).expand(-1, -1, -1, self.n_query, -1)
            value_attn = value_states.unsqueeze(3).expand(-1, -1, -1, self.n_query, -1)
            key_attn = key_attn.reshape(B, T, self.total_heads, head_dim)
            value_attn = value_attn.reshape(B, T, self.total_heads, head_dim)
        else:
            key_attn = key_states.reshape(B, T, self.total_heads, head_dim)
            value_attn = value_states.reshape(B, T, self.total_heads, head_dim)
        q = query_states.transpose(1, 2)
        v = value_attn.transpose(1, 2)
        k_new = key_attn.transpose(1, 2)
        base_k = key_states.transpose(1, 2)
        base_v = value_states.transpose(1, 2)
        key_append = base_k.squeeze(2).unsqueeze(2)
        value_append = base_v.squeeze(2).unsqueeze(2)
        if write_cache:
            cache.append(key_append, value_append)
            k, v = cache.tensors()
        else:
            k, v = key_append, value_append
        if self.n_query > 1:
            k = k.repeat_interleave(self.n_query, dim=1)
            v = v.repeat_interleave(self.n_query, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        if puncture_mask is not None:
            att = att.masked_fill(puncture_mask[:, None, None, :], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.total_heads * head_dim)
        if disable_rows is not None and disable_rows.any():
            row_mask = (~disable_rows).view(-1, 1, 1).to(y.dtype)
            y = y * row_mask
        return self.proj(y), cache


class FeedForward(nn.Module):
    """Position-wise MLP reused by every :class:`Block`."""

    def __init__(self, args: Args) -> None:
        super().__init__()
        config = args
        hidden = 4 * config.n_width
        self.fc1 = nn.Linear(config.n_width, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, config.n_width)
        self.drop = nn.Dropout(config.dropout)
        self.use_gmlp = bool(getattr(config, "use_gmlp", False))
        self.fc_gate = nn.Linear(config.n_width, hidden) if self.use_gmlp else None

    def forward(
        self, x: torch.Tensor, *, record_mask: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.fc1(x)
        mask = None
        if record_mask:
            mask = hidden[:, -1, :] > 0
        activated = self.act(hidden)
        if self.use_gmlp and self.fc_gate is not None:
            gate = self.fc_gate(x)
            activated = activated * gate
        out = self.fc2(activated)
        out = self.drop(out)
        return out, mask


class Block(nn.Module):
    """Transformer block consumed by :class:`TransformerStackCore` and :class:`GPTCore`."""

    def __init__(self, args: Args) -> None:
        super().__init__()
        config = args
        self.ln1 = nn.LayerNorm(config.n_width)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_width)
        self.ff = FeedForward(config)
        self.xctx_attn_gain = nn.Parameter(torch.ones(config.n_width))
        self.xctx_mlp_gain = nn.Parameter(torch.ones(config.n_width))
        self.grce_attn_ld = LayerDampening(config.n_width)
        self.grce_mlp_ld = LayerDampening(config.n_width)

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
        rope_positions: torch.Tensor | None = None,
        position_offsets: torch.Tensor | None = None,
        sane_hook=None,
        sane_group_ids: torch.Tensor | None = None,
        sane_z_indices: torch.Tensor | None = None,
        sane_active_mask: torch.Tensor | None = None,
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
            rope_positions=rope_positions,
            position_offsets=position_offsets,
            attention_capture=attention_capture,
            sane_group_ids=sane_group_ids,
            sane_z_indices=sane_z_indices,
            sane_active_mask=sane_active_mask,
        )
        if attention_disabled_rows is not None and attention_disabled_rows.any():
            mask = (~attention_disabled_rows).view(-1, 1, 1).to(attn_output.dtype)
            attn_output = attn_output * mask
        x = x + attn_output
        if sane_hook is not None:
            sane_hook(layer_idx, "attn", attn_output, x)
        ff_input = self._apply_xctx_bias(x, xctx_bias, self.xctx_mlp_gain)
        pre_ff = self.ln2(ff_input)
        pre_ff = self._apply_grce_bias(pre_ff, grce_bias, self.grce_mlp_ld)
        ff_out, mask = self.ff(pre_ff, record_mask=record_mask)
        x = x + ff_out
        if sane_hook is not None:
            sane_hook(layer_idx, "mlp", ff_out, x)
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
        column_position: int | None = None,
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
            position_index=column_position,
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
    n_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Sum a list of optional bias tensors into a single tensor."""

    if not bias_list:
        return None
    merged = torch.zeros(rows, cols, n_layers, n_width, device=device, dtype=dtype)
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
    """Collapse many small cache segments into a single chunk per layer.

       The implemented method is based on the following assumptions:
       - Merging two caches of approx same size (same .bit_length()) is always good
       - When creating a merged cache it is always good to merge anything smaller into it too
       - The order of caches does not matter and empty caches can be dropped entirely
    """

    if not kv_cache_list or (kv_cache_list[0] and kv_cache_list[0][0] and
                             len(kv_cache_list) <= kv_cache_list[0][0][0].size(1).bit_length()):
        return kv_cache_list

    sorted_caches = list()
    clog2_counts = defaultdict(int)
    for kv_cache in kv_cache_list:
        if not kv_cache or not kv_cache[0]: continue
        n_kv = kv_cache[0][0].size(1)
        if not n_kv: continue
        sorted_caches.append((n_kv, kv_cache))
        clog2_n_kv = n_kv.bit_length()
        clog2_counts[clog2_n_kv] += 1

    sorted_caches = sorted(sorted_caches, key=lambda item: item[0])

    clog2_filtered_list = [clog2_n_kv for clog2_n_kv, count in clog2_counts.items() if count > 1]
    clog2_cursor = max(clog2_filtered_list) if clog2_filtered_list else 0

    nomerge_caches = list()
    merge_caches = list()
    merge_n_kv = 0

    for n_kv, kv_cache in sorted_caches:
        if n_kv.bit_length() > clog2_cursor:
            nomerge_caches.append(kv_cache)
        else:
            merge_caches.append(kv_cache)
            merge_n_kv += n_kv
            clog2_cursor = max(clog2_cursor, merge_n_kv.bit_length())

    final_caches = list(reversed(nomerge_caches))
    if merge_caches:
        final_caches.append(kv_cache_list_merge(merge_caches))
    return final_caches


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
        if config.n_width % 2 != 0:
            raise ValueError("n_width must be even so embeddings can occupy even/odd slots")
        self.embedding_dim = config.n_width // 2
        self.tok_emb = nn.Embedding(config.vocab_size, self.embedding_dim)
        self.control_emb = nn.Embedding(CONTROL_EMBEDDING_ROWS, self.embedding_dim, padding_idx=0)
        self.think_emb = ThinkEmbeddingLibrary(self.embedding_dim, think_spans=(2, 3, 4))
        self.loop_embeddings = nn.ParameterDict(
            {
                str(count): nn.Parameter(torch.zeros(self.embedding_dim))
                for count in (2, 3, 4)
            }
        )
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_width)
        self.loop_ln = nn.LayerNorm(config.n_width)
        self.sane_loop_norm = RMSNormNoAffine(config.n_width)
        self.head = nn.Linear(self.embedding_dim, config.vocab_size, bias=False)
        self.use_rope_xl = bool(getattr(config, "use_rope_xl", False))
        self.use_sane = bool(getattr(config, "use_sane", False))
        self.rope_xl_half_window = float(config.n_pos) / 2.0 if self.use_rope_xl else 0.0
        self.sane_stream_width = config.n_width // 2 if self.use_sane else 0
        if self.use_sane:
            half = self.sane_stream_width
            self.sane_alpha_attn = nn.Parameter(torch.zeros(config.n_layer, half))
            self.sane_beta_attn = nn.Parameter(torch.zeros(config.n_layer, half))
            self.sane_alpha_mlp = nn.Parameter(torch.zeros(config.n_layer, half))
            self.sane_beta_mlp = nn.Parameter(torch.zeros(config.n_layer, half))

    def expand_to_even(self, tensor: torch.Tensor) -> torch.Tensor:
        """Place half-width embedding features into the even data-path slots."""

        return _expand_tensor_to_parity(tensor, self.config.n_width, parity=0)

    def output_features(self, tensor: torch.Tensor, *, use_next_stream: bool = True) -> torch.Tensor:
        """Extract either the next-stream (odd) or self-stream (even) slots."""

        parity = 1 if use_next_stream else 0
        return _select_parity_features(tensor, parity=parity)

    @staticmethod
    def swap_self_next_streams(tensor: torch.Tensor) -> torch.Tensor:
        even = tensor[..., ::2]
        odd = tensor[..., 1::2]
        swapped = torch.empty_like(tensor)
        swapped[..., ::2] = odd
        swapped[..., 1::2] = even
        return swapped

    def _sane_stage_params(self, stage: str) -> tuple[torch.Tensor, torch.Tensor]:
        if stage == "attn":
            return self.sane_alpha_attn, self.sane_beta_attn
        if stage == "mlp":
            return self.sane_alpha_mlp, self.sane_beta_mlp
        raise ValueError(f"Unknown SANE stage: {stage}")

    def _apply_sane_propagation(
        self,
        *,
        delta: torch.Tensor,
        tensor: torch.Tensor,
        layer_idx: int,
        stage: str,
        mode: str,
        sane_first_columns: torch.Tensor | None,
        sane_group_ids: torch.Tensor | None,
        sane_z_indices: torch.Tensor | None,
        sane_active_mask: torch.Tensor | None,
    ) -> None:
        if not self.use_sane or delta is None:
            return
        if tensor.size(1) <= 1:
            return
        alpha_table, beta_table = self._sane_stage_params(stage)
        half = self.sane_stream_width
        if half <= 0:
            return
        alpha = alpha_table[layer_idx].view(1, 1, half)
        beta = beta_table[layer_idx].view(1, 1, half)
        next_stream = delta[..., 1::2]
        self_stream = delta[..., ::2]
        group_ids = None
        alpha_adj_mask = None
        if sane_group_ids is not None or sane_z_indices is not None:
            with torch.no_grad():
                if sane_group_ids is not None:
                    group_ids = sane_group_ids.to(delta.device).clone()
                if sane_z_indices is not None:
                    z_ids = sane_z_indices.to(delta.device).clone()
                    if z_ids.numel() >= 2:
                        alpha_adj_mask = (z_ids[1:] == (z_ids[:-1] + 1)).view(1, -1, 1)
        if mode in {"decode", "encode"}:
            addition = next_stream[:, :-1, :] * alpha
            if alpha_adj_mask is not None:
                addition = addition * alpha_adj_mask.to(addition.dtype)
            tensor[:, 1:, ::2] += addition
            if sane_first_columns is not None and sane_first_columns.numel() > 1:
                indices = sane_first_columns.to(delta.device)
                src = indices[:-1]
                dst = indices[1:]
                addition = next_stream[:, src, :] * alpha
                tensor[:, dst, ::2] += addition
        apply_beta = False
        group_mask = None
        if mode in {"reverse", "encode"}:
            apply_beta = True
        elif mode == "decode" and group_ids is not None:
            apply_beta = True
            if group_ids.numel() >= 2:
                with torch.no_grad():
                    group_mask = (group_ids[1:] == group_ids[:-1]).view(1, -1, 1)
        edge_active_mask = None
        if sane_active_mask is not None:
            active_vec = sane_active_mask.to(delta.device, dtype=torch.bool)
            if active_vec.dim() != 1 or active_vec.size(0) != tensor.size(1):
                raise ValueError("sane_active_mask must match column count")
            if active_vec.numel() >= 2:
                edge_active_mask = (active_vec[1:] & active_vec[:-1]).view(1, -1, 1)
        if apply_beta:
            addition = self_stream[:, 1:, :] * beta
            if group_mask is not None:
                addition = addition * group_mask.to(addition.dtype)
            if edge_active_mask is not None:
                addition = addition * edge_active_mask.to(addition.dtype)
            tensor[:, :-1, 1::2] += addition

    def loop_embedding(self, repeat: int) -> torch.Tensor | None:
        key = str(int(repeat))
        if key not in self.loop_embeddings:
            return None
        return self.loop_embeddings[key]

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
        rope_positions: torch.Tensor | None = None,
        attention_capture: AttentionCapture | None = None,
        layer_repeat: int = 1,
        layer_top_only: bool = False,
        enable_sane: bool | None = None,
        sane_first_columns: torch.Tensor | None = None,
        sane_group_ids: torch.Tensor | None = None,
        sane_z_indices: torch.Tensor | None = None,
        sane_active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        rows, cols, _ = x.shape
        rope_positions_tensor = None
        if rope_positions is not None:
            rope_positions_tensor = rope_positions.to(device=x.device, dtype=torch.long)
            if rope_positions_tensor.dim() != 1 or rope_positions_tensor.size(0) != cols:
                raise ValueError("rope_positions must match column count")
        if rope_positions_tensor is None:
            rope_positions_tensor = torch.arange(cols, dtype=torch.long, device=x.device)
        position_offsets_tensor = None
        if position_offsets is not None:
            position_offsets_tensor = position_offsets.to(device=x.device, dtype=torch.long)
            if position_offsets_tensor.dim() != 1 or position_offsets_tensor.size(0) != rows:
                raise ValueError("position_offsets must match row count")
        xctx_tensor = _merge_bias_list(
            xctx_bias_list_in or [],
            rows,
            cols,
            self.config.n_layer,
            self.config.n_width,
            x.device,
            x.dtype,
        )
        grce_tensor = _merge_bias_list(
            grce_bias_list_in or [],
            rows,
            cols,
            self.config.n_layer,
            self.config.n_width,
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
        repeats = max(1, int(layer_repeat))
        samples: list[torch.Tensor | None] = [None] * len(self.blocks)
        kv_outputs: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.blocks)
        sane_enabled = self.use_sane and mode in {"decode", "reverse", "encode"}
        if enable_sane is not None:
            sane_enabled = bool(enable_sane)
        sane_hook = None
        if sane_enabled:
            def sane_hook(layer_idx: int, stage: str, delta: torch.Tensor, tensor: torch.Tensor) -> None:
                self._apply_sane_propagation(
                    delta=delta,
                    tensor=tensor,
                    layer_idx=layer_idx,
                    stage=stage,
                    mode=mode,
                    sane_first_columns=sane_first_columns,
                    sane_group_ids=sane_group_ids,
                    sane_z_indices=sane_z_indices,
                    sane_active_mask=sane_active_mask,
                )
        for rep_idx in range(repeats):
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
                    rope_positions=rope_positions_tensor,
                    position_offsets=position_offsets_tensor,
                    sane_hook=sane_hook,
                    sane_group_ids=sane_group_ids,
                    sane_z_indices=sane_z_indices,
                    sane_active_mask=sane_active_mask,
                )
                samples[layer_idx] = current
                if kv_pair is None:
                    if layer_top_only and rep_idx == repeats - 1:
                        kv_outputs[layer_idx] = None
                    continue
                existing = kv_outputs[layer_idx]
                if layer_top_only:
                    if rep_idx == repeats - 1 or existing is None:
                        kv_outputs[layer_idx] = kv_pair
                else:
                    if existing is None:
                        kv_outputs[layer_idx] = kv_pair
                    else:
                        key = torch.cat([existing[0], kv_pair[0]], dim=1)
                        value = torch.cat([existing[1], kv_pair[1]], dim=1)
                        kv_outputs[layer_idx] = (key, value)
        finalized_samples: list[torch.Tensor] = []
        for tensor in samples:
            if tensor is None:
                raise RuntimeError("Missing layer sample; layer_repeat loop malfunctioned")
            finalized_samples.append(tensor)
        return current, finalized_samples, kv_outputs


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
        layer_repeat: int = 1,
        layer_top_only: bool = False,
        position_offsets: torch.Tensor | None = None,
        rope_positions: torch.Tensor | None = None,
        attention_capture: AttentionCapture | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list]:
        output, samples, kv_out = self.core.forward_grid(
            x,
            xctx_bias_list_in=xctx_bias_list_in,
            grce_bias_list_in=grce_bias_list_in,
            kv_cache_list_in=kv_cache_list_in,
            mode=mode,
            qh_query_callback=qh_query_callback,
            position_offsets=position_offsets,
            rope_positions=rope_positions,
            attention_capture=attention_capture,
            layer_repeat=layer_repeat,
            layer_top_only=layer_top_only,
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
        self.n_width = config.n_width
        self.detach_span = max(0, int(config.detach_span))
        self.detach_ctx_enabled_default = not getattr(config, "no_detach_ctx", False)
        if self.disabled:
            return
        self.sample_norms = nn.ModuleList(
            nn.LayerNorm(self.n_width) for _ in range(self.n_layers)
        )
        self.sample_projections = nn.ModuleList(
            nn.Linear(self.n_width, self.context_dim) for _ in range(self.n_layers)
        )
        self.bias_norm = nn.LayerNorm(self.context_dim)
        self.bias_projections = nn.ModuleList(
            nn.Linear(self.context_dim, self.n_width) for _ in range(self.n_layers)
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
            return grce_state.new_zeros(grce_state.size(0), 1, 0, self.n_width)
        normed = self.bias_norm(grce_state)
        per_layer = [proj(normed) for proj in self.bias_projections]
        stacked = torch.stack(per_layer, dim=1)
        return stacked.unsqueeze(1)

    def sample_forward(
        self,
        grce_state: torch.Tensor,
        samples: Sequence[torch.Tensor],
        column_index: int,
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
        selected: list[torch.Tensor] = []
        for layer_idx in range(self.n_layers):
            layer_sample = samples[layer_idx]
            if layer_sample.size(1) == 0:
                raise RuntimeError("Layer sample tensor has zero length")
            idx = min(column_index, layer_sample.size(1) - 1)
            selected.append(layer_sample[:, idx, :])
        layer_stack = torch.stack(selected, dim=0)
        if should_detach:
            layer_stack = layer_stack.detach()

        eps = self.sample_norms[0].eps if self.sample_norms else 1e-5
        weight = torch.stack([norm.weight for norm in self.sample_norms], dim=0)
        bias = torch.stack([norm.bias for norm in self.sample_norms], dim=0)
        mean = layer_stack.mean(dim=-1, keepdim=True)
        var = layer_stack.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (layer_stack - mean) * torch.rsqrt(var + eps)
        normed = normalized * weight.unsqueeze(1) + bias.unsqueeze(1)

        proj_weight = torch.stack([linear.weight for linear in self.sample_projections], dim=0)
        proj_bias = torch.stack([linear.bias for linear in self.sample_projections], dim=0)
        projected = torch.matmul(normed, proj_weight.transpose(-1, -2)) + proj_bias.unsqueeze(1)
        fused = projected.sum(dim=0)
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
        self.n_width = config.n_width
        self.context_dim = config.n_xctx
        self.inner_dim = _get_inner_xctx_width(config)
        self.squeeze_dim = max(1, self.context_dim // 2)
        self.detach_span = max(0, int(config.detach_span))
        self.detach_ctx_enabled_default = not getattr(config, "no_detach_ctx", False)
        if self.disabled:
            return
        self.sample_linear = nn.ModuleList(
            nn.Linear(self.n_width, self.inner_dim) for _ in range(self.n_layers)
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
            nn.Linear(self.inner_dim, self.n_width) for _ in range(self.n_layers)
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
            return xctx_state.new_zeros(xctx_state.size(0), 1, 0, self.n_width)
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
        column_index: int,
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
        selected: list[torch.Tensor] = []
        for idx in range(self.n_layers):
            layer_sample = samples[idx]
            if layer_sample.size(1) == 0:
                raise RuntimeError("Layer sample tensor has zero length")
            col_idx = min(column_index, layer_sample.size(1) - 1)
            selected.append(layer_sample[:, col_idx, :])
        layer_stack = torch.stack(selected, dim=0)
        if should_detach:
            layer_stack = layer_stack.detach()
        centered = layer_stack - layer_stack.mean(dim=-1, keepdim=True)

        lin_weight = torch.stack([linear.weight for linear in self.sample_linear], dim=0)
        lin_bias = torch.stack([linear.bias for linear in self.sample_linear], dim=0)
        reduced = torch.matmul(centered, lin_weight.transpose(-1, -2)) + lin_bias.unsqueeze(1)

        eps = self.sample_norms[0].eps if self.sample_norms else 1e-5
        norm_weight = torch.stack([norm.weight for norm in self.sample_norms], dim=0)
        norm_bias = torch.stack([norm.bias for norm in self.sample_norms], dim=0)
        mean = reduced.mean(dim=-1, keepdim=True)
        var = reduced.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (reduced - mean) * torch.rsqrt(var + eps)
        normed = normalized * norm_weight.unsqueeze(1) + norm_bias.unsqueeze(1)

        expand_weight = torch.stack([linear.weight for linear in self.expand_linear], dim=0)
        expand_bias = torch.stack([linear.bias for linear in self.expand_linear], dim=0)
        expanded = torch.matmul(normed, expand_weight.transpose(-1, -2)) + expand_bias.unsqueeze(1)
        fused = expanded.sum(dim=0)
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
        column_position: int | None = None,
        layer_repeat: int = 1,
        layer_top_only: bool = False,
        use_context: bool = True,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        if use_context:
            grce_biases = self._collect_biases(grce_bias_list_in, column_index)
            xctx_biases = self._collect_biases(xctx_bias_list_in, column_index)
            if self.grce is not None and grce_state is not None:
                grce_biases.append(self.grce.bias_forward(grce_state))
            if self.xctx is not None and xctx_state is not None:
                xctx_biases.append(self.xctx.bias_forward(xctx_state))
        else:
            grce_biases = []
            xctx_biases = []
        rope_positions = None
        if column_position is not None:
            rope_positions = torch.tensor(
                [column_position], device=column_input.device, dtype=torch.long
            )
        column_output, samples, kv_pairs = self.core.forward_grid(
            column_input,
            xctx_bias_list_in=xctx_biases,
            grce_bias_list_in=grce_biases,
            kv_cache_list_in=kv_sources,
            mode="decode",
            qh_query_callback=qh_query_callback,
            rope_positions=rope_positions,
            attention_capture=attention_capture,
            layer_repeat=layer_repeat,
            layer_top_only=layer_top_only,
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
        self.n_width = config.n_width
        self.grce = TransformerGRCE(config) if config.n_grce > 0 else None
        self.xctx = TransformerXCTX(config) if config.n_xctx > 0 else None
        self.kv_rebalance = args.kv_rebalance
        self.detach_think_span = max(0, int(getattr(config, "detach_think_span", 0)))
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
        head_dim = self.n_width // n_heads
        storage: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(self.n_layers):
            key_buf = torch.empty(rows, cols, n_heads, head_dim, device=device, dtype=dtype)
            value_buf = torch.empty_like(key_buf)
            storage.append((key_buf, value_buf))
        return storage

    def _detached_kv_prefix(
        self,
        storage: list[tuple[torch.Tensor, torch.Tensor]],
        start_col: int,
        end_col: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        if end_col <= start_col:
            return [None] * len(storage)
        prefix: list[tuple[torch.Tensor, torch.Tensor]] = []
        for key_buf, value_buf in storage:
            prefix.append((key_buf[:, start_col:end_col, :, :], value_buf[:, start_col:end_col, :, :]))
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
        think_step_index: torch.Tensor | None = None,
        think_step_count: torch.Tensor | None = None,
        column_positions: torch.Tensor | None = None,
        sane_first_columns: torch.Tensor | None = None,
        sane_group_ids: torch.Tensor | None = None,
        sane_z_indices: torch.Tensor | None = None,
        sane_active_mask: torch.Tensor | None = None,
        layer_repeat: int = 1,
        layer_top_only: bool = False,
        think_last_only: bool = False,
        capture_layer_outputs: bool = False,
        use_context: bool = True,
        disable_sane: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list, list[torch.Tensor] | None]:
        if mode not in {"forward", "encode", "decode", "reverse", "noattn"}:
            raise ValueError(f"Unknown TransformerStackSequence mode: {mode}")
        rows, cols, _ = x.shape
        device = x.device
        dtype = x.dtype
        kv_repeat_factor = layer_repeat if (layer_repeat > 1 and not layer_top_only) else 1
        outputs: list[torch.Tensor] = []
        captured_layers: list[list[torch.Tensor]] | None = None
        if capture_layer_outputs:
            captured_layers = [[] for _ in range(self.n_layers)]
        grce_state = self._ensure_state(self.grce, grce_in, rows, device, dtype)
        xctx_state = self._ensure_state(self.xctx, xctx_in, rows, device, dtype)
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
                column_positions=column_positions,
                sane_first_columns=sane_first_columns,
                sane_group_ids=sane_group_ids,
                sane_z_indices=sane_z_indices,
                sane_active_mask=sane_active_mask,
                layer_repeat=layer_repeat,
                layer_top_only=layer_top_only,
                think_last_only=think_last_only,
                capture_layer_outputs=capture_layer_outputs,
                use_context=use_context,
                disable_sane=disable_sane,
            )
        column_positions_tensor = None
        if column_positions is not None:
            positions = column_positions.to(device=device, dtype=torch.long)
            if positions.dim() != 1 or positions.size(0) != cols:
                raise ValueError("column_positions must match sequence columns")
            column_positions_tensor = positions
        use_internal_cache = mode != "noattn"
        base_sources = list(kv_cache_list_in or [])
        kv_history: list[list[tuple[torch.Tensor, torch.Tensor] | None]] = []
        kv_storage: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        kv_storage_cursor = 0
        kv_storage_visible_start = 0
        kv_storage_lengths: list[int] = []
        if detach_internal_kv_cache:
            kv_storage = self._allocate_detached_kv_storage(
                rows,
                cols * kv_repeat_factor,
                device,
                dtype,
            )
        loop_residual: torch.Tensor | None = None
        think_group_size = self.detach_think_span
        think_detach_active = (
            think_group_size > 0
            and think_step_index is not None
            and think_step_count is not None
        )
        think_span_counter = 0

        def _detach_think_boundary() -> None:
            nonlocal grce_state, xctx_state, base_sources, kv_history
            if grce_state is not None:
                grce_state = grce_state.detach()
            if xctx_state is not None:
                xctx_state = xctx_state.detach()
            if base_sources:
                base_sources = kv_cache_list_detach(base_sources)
            if (not detach_internal_kv_cache) and kv_history:
                kv_history = kv_cache_list_detach(kv_history)

        for col in range(cols):
            column_input = x[:, col : col + 1, :]
            column_position = None
            if column_positions_tensor is not None:
                column_position = int(column_positions_tensor[col].item())
            if think_step_index is not None and think_step_count is not None:
                step_index = int(think_step_index[col].item())
                step_count = int(think_step_count[col].item())
            else:
                step_index = 1
                step_count = 1
            is_think_column = think_detach_active and step_count > 1
            if step_index <= 1:
                loop_residual = None
            if loop_residual is not None:
                column_input = column_input + loop_residual
            if think_detach_active and is_think_column and step_index == 1:
                think_span_counter += 1
                span_offset = (think_span_counter - 1) % think_group_size
                if span_offset == 0:
                    _detach_think_boundary()
            elif think_detach_active and not is_think_column:
                think_span_counter = 0
            column_kv_sources: list[Sequence[tuple[torch.Tensor, torch.Tensor] | None]] = []
            if base_sources:
                column_kv_sources.extend(base_sources)
            if detach_internal_kv_cache and kv_storage is not None and kv_storage_cursor > kv_storage_visible_start:
                column_kv_sources.append(
                    self._detached_kv_prefix(
                        kv_storage,
                        kv_storage_visible_start,
                        kv_storage_cursor,
                    )
                )
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
                column_position=column_position,
                layer_repeat=layer_repeat,
                layer_top_only=layer_top_only,
                use_context=use_context,
            )
            outputs.append(column_output)
            if captured_layers is not None:
                for layer_idx in range(min(len(samples), self.n_layers)):
                    captured_layers[layer_idx].append(samples[layer_idx])
            if step_count > 1 and step_index < step_count:
                loop_residual = self.core.loop_ln(column_output)
            else:
                loop_residual = None
            kv_column_length = _kv_column_length(kv_pairs)
            if detach_internal_kv_cache and kv_storage is not None and kv_column_length > 0:
                for layer_idx, kv_pair in enumerate(kv_pairs):
                    if kv_pair is None:
                        continue
                    key_chunk, value_chunk = kv_pair
                    key_buf, value_buf = kv_storage[layer_idx]
                    kv_start = kv_storage_cursor
                    kv_end = kv_start + key_chunk.size(1)
                    key_buf[:, kv_start:kv_end, :, :].copy_(key_chunk.detach())
                    value_buf[:, kv_start:kv_end, :, :].copy_(value_chunk.detach())
                kv_storage_cursor += kv_column_length
                kv_storage_lengths.append(kv_column_length)
            elif detach_internal_kv_cache and kv_storage is not None:
                kv_storage_lengths.append(0)
            else:
                kv_history.append(kv_pairs)
                if self.kv_rebalance and not think_last_only:
                    kv_history = kv_cache_list_balance(kv_history)
            detach_samples = detach_samples_span > 0 and (col % detach_samples_span) == 0
            if use_context and self.grce is not None and grce_state is not None:
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
            if use_context and self.xctx is not None and xctx_state is not None:
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
            if (
                think_last_only
                and think_step_index is not None
                and think_step_count is not None
                and step_count > 1
                and step_index == step_count
            ):
                drop_columns = max(0, step_count - 1)
                if drop_columns > 0:
                    _kv_history_prune_last(kv_history, drop_columns)
                    if detach_internal_kv_cache and kv_storage is not None:
                        drop_len = _kv_storage_prune_lengths(kv_storage_lengths, drop_columns)
                        kv_storage_visible_start += drop_len
            if think_detach_active and is_think_column and step_index == step_count:
                next_is_think = False
                if (col + 1) < cols:
                    next_step_count = int(think_step_count[col + 1].item())
                    next_is_think = next_step_count > 1
                group_closed = (think_span_counter % think_group_size) == 0
                if group_closed or (not next_is_think):
                    _detach_think_boundary()
                if not next_is_think:
                    think_span_counter = 0
        stacked = torch.cat(outputs, dim=1)
        if detach_internal_kv_cache and kv_storage is not None:
            length = kv_storage_cursor - kv_storage_visible_start
            kv_out: list[tuple[torch.Tensor, torch.Tensor] | None] = []
            for key_buf, value_buf in kv_storage:
                if length <= 0:
                    kv_out.append(None)
                    continue
                key_slice = key_buf[
                    :, kv_storage_visible_start:kv_storage_cursor, :, :
                ].detach()
                value_slice = value_buf[
                    :, kv_storage_visible_start:kv_storage_cursor, :, :
                ].detach()
                kv_out.append((key_slice, value_slice))
        else:
            kv_out = kv_cache_list_merge(kv_history)
        layer_outputs = None
        if captured_layers is not None:
            layer_outputs = [
                torch.cat(chunks, dim=1) if chunks else stacked.new_zeros(rows, 0, self.n_width)
                for chunks in captured_layers
            ]
        return stacked, grce_state, xctx_state, kv_out, layer_outputs

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
        column_positions: torch.Tensor | None,
        sane_first_columns: torch.Tensor | None,
        sane_group_ids: torch.Tensor | None,
        sane_z_indices: torch.Tensor | None,
        sane_active_mask: torch.Tensor | None,
        layer_repeat: int,
        layer_top_only: bool,
        think_last_only: bool,
        capture_layer_outputs: bool,
        use_context: bool = True,
        disable_sane: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        list[tuple[torch.Tensor, torch.Tensor]],
        list[torch.Tensor] | None,
    ]:
        rows, cols, _ = x.shape
        rope_positions = None
        if column_positions is not None:
            rope_positions = column_positions.to(device=x.device, dtype=torch.long)
            if rope_positions.dim() != 1 or rope_positions.size(0) != cols:
                raise ValueError("column_positions must match sequence columns")
        grid_grce_biases = [] if not use_context else list(grce_bias_list_in or [])
        grid_xctx_biases = [] if not use_context else list(xctx_bias_list_in or [])
        if use_context and self.grce is not None and grce_state is not None:
            grid_grce_biases.append(self.grce.bias_forward(grce_state))
        if use_context and self.xctx is not None and xctx_state is not None:
            grid_xctx_biases.append(self.xctx.bias_forward(xctx_state))
        sane_override = False if disable_sane else None
        output, samples, kv_pairs = self.core.forward_grid(
            x,
            xctx_bias_list_in=grid_xctx_biases,
            grce_bias_list_in=grid_grce_biases,
            kv_cache_list_in=kv_cache_list_in,
            mode=mode,
            qh_query_callback=qh_query_callback,
            rope_positions=rope_positions,
            attention_capture=attention_capture,
            layer_repeat=layer_repeat,
            layer_top_only=layer_top_only,
            enable_sane=sane_override,
            sane_first_columns=sane_first_columns,
            sane_group_ids=sane_group_ids,
            sane_z_indices=sane_z_indices,
            sane_active_mask=sane_active_mask,
        )
        kv_out = kv_pairs
        layer_outputs = samples if capture_layer_outputs else None
        if detach_internal_kv_cache:
            kv_out = [
                None if pair is None else (pair[0].detach(), pair[1].detach())
                for pair in kv_out
            ]
        detach_samples = False
        for col in range(cols):
            if detach_samples_span > 0:
                detach_samples = (col % detach_samples_span) == 0
            if use_context and self.grce is not None and grce_state is not None:
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
            if use_context and self.xctx is not None and xctx_state is not None:
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
        return output, grce_state, xctx_state, kv_out, layer_outputs


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
        if torch.any(pos_idx >= self.config.n_pos):
            raise ValueError("position ids exceed configured --n-pos")
        tok = self.core.expand_to_even(self.core.tok_emb(idx))
        x = self.core.drop(tok)
        context_info: dict[str, torch.Tensor] | None = None
        if mode in {"forward", "noattn", "encode"}:
            sequence_output, grce_out, xctx_out, _, _ = self.stack_sequence.forward(
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
        elif mode in {"decode", "reverse"}:
            decode_output, _, _ = self.stack_grid.forward(x, mode=mode)
            hidden = decode_output
        else:
            raise ValueError(f"Unknown forward_autoreg mode: {mode}")
        head_features = self.core.ln_f(hidden)
        use_next_stream = mode != "reverse"
        logits = self.core.head(
            self.core.output_features(head_features, use_next_stream=use_next_stream)
        )
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
        f"v{config.vocab_size}_n{config.n_pos}_w{config.n_width}_"
        f"d{config.n_layer}_h{config.n_head}"
    )
    if config.n_grce > 0:
        tag += f"_g{config.n_grce}"
    if config.n_xctx > 0:
        tag += f"_x{config.n_xctx}"
    if getattr(config, "n_query", 1) != 1:
        tag += f"_q{config.n_query}"
    if getattr(config, "use_gmlp", False):
        tag += "_gmlp"
    if getattr(config, "use_rope_xl", False):
        tag += "_ropex"
    if getattr(config, "use_rope_vr", False):
        tag += "_ropevr"
    if getattr(config, "use_rope_vr_all", False):
        tag += "_ropevrall"
    if getattr(config, "use_sane", False):
        tag += "_sane"
    return tag


LOSS_IGNORE_INDEX = -100


# -----------------------------------------------------------------------------
# Training / Generation Helpers
# -----------------------------------------------------------------------------


BATCH_MODES: tuple[str, ...] = ("encode", "decode", "forward", "think", "noattn", "reverse")
METRIC_BUCKET_ORDER = ["target", *BATCH_MODES]


ROW_METRIC_HIST_KEYS = list(BATCH_MODES)
ROW_METRIC_LOG_KEYS = list(BATCH_MODES)
ROW_METRIC_LOG_GROUP = {"encode", "forward", "noattn"}


@dataclass
class EvalBatchStats:
    """Holds averaged losses plus raw sums/counts for evaluation batches."""

    metrics: dict[str, float | None]
    loss_sums: dict[str, float]
    token_counts: dict[str, int]


def _combine_eval_stats(stats_list: Sequence[EvalBatchStats]) -> EvalBatchStats:
    """Merge multiple evaluation runs by summing loss totals and counts."""

    key_set: set[str] = {"target"}
    for stats in stats_list:
        key_set.update(stats.loss_sums.keys())
    ordered_keys = list(BATCH_MODES)
    extras = sorted(key for key in key_set if key not in set(ordered_keys) | {"target"})
    ordered_keys += extras + ["target"]
    combined_sums = {key: 0.0 for key in ordered_keys}
    combined_counts = {key: 0 for key in ordered_keys}
    for stats in stats_list:
        for key in ordered_keys:
            combined_sums[key] += stats.loss_sums.get(key, 0.0)
            combined_counts[key] += stats.token_counts.get(key, 0)
    metrics = {key: None for key in ordered_keys}
    for key in ordered_keys:
        count = combined_counts[key]
        if count > 0:
            metrics[key] = combined_sums[key] / count
    return EvalBatchStats(metrics, combined_sums, combined_counts)


@dataclass
class LayoutPassResult:
    total_loss_sum: torch.Tensor | None
    total_tokens: int
    base_tokens: int
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


def _row_future_margin(row: BlockLayout) -> int:
    margin = 0
    for segment in row.segments:
        if segment.mode != "decode":
            continue
        depth = max(1, int(getattr(segment, "sane_z", 1) or 1))
        margin = max(margin, max(0, depth - 2))
    return margin


def _sane_column_plan(base_tokens: int, depth: int) -> list[tuple[int, int]]:
    if base_tokens <= 0:
        return []
    depth = max(1, depth)
    plan: list[tuple[int, int]] = []
    span = max(0, base_tokens - 1)
    for idx in range(span):
        for z in range(depth):
            plan.append((idx, z))
    plan.append((base_tokens - 1, 0))
    return plan


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
    base_token_total = 0
    for group in rows:
        row_count = int(group.rows)
        if row_count <= 0:
            continue
        cols_total = group.total_columns()
        pos_total = group.total_positions()
        if cols_total <= 0:
            continue
        if pos_total <= 0:
            continue
        modifiers = group.modifiers
        detach_span_override = (
            modifiers.detach_span if modifiers and modifiers.detach_span is not None else None
        )
        row_detach_kv_cache = args.detach_kv_cache or (
            modifiers.detach_kv_cache if modifiers else False
        )
        context_detach_override = False if (modifiers and modifiers.no_detach_ctx) else None
        row_disable_sane = bool(modifiers and getattr(modifiers, "disable_sane", False))
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
                future_margin = _row_future_margin(group)
                xb_base, yb_base, metadata, future_tokens = dataset.sample_row_batch(
                    split,
                    pos_total,
                    row_count,
                    device,
                    rng=rng,
                    future_margin=future_margin,
                )
                base_token_total += int(xb_base.numel())
            except ValueError as exc:
                raise ValueError(
                    f"Unable to sample {row_count} rows with {pos_total} tokens from the corpus"
                ) from exc
            yb_extended = yb_base if future_tokens.size(1) == 0 else torch.cat(
                [yb_base, future_tokens], dim=1
            )
            xb = _expand_think_sequences(xb_base, group.segments)
            yb = _expand_think_sequences(yb_base, group.segments)
            pos_offsets = None
            if position_shift:
                pos_offsets = torch.full((row_count,), position_shift, dtype=torch.long, device=device)
            token_components = _token_embeddings_with_offsets(
                model,
                xb,
                pos_offsets,
            )
            future_token_components = _token_embeddings_with_offsets(
                model,
                yb,
                pos_offsets,
            )
            column_positions = _segment_column_positions(group.segments, device)
            if group.modifiers and getattr(group.modifiers, "halt_rope", False):
                column_positions = torch.zeros_like(column_positions)
            control_ids = torch.zeros((row_count, cols_total), dtype=torch.long, device=device)
            cursor = 0
            pos_cursor = 0
            kv_chain: list[list[tuple[torch.Tensor, torch.Tensor]] | None] = []
            grce_state = None
            xctx_state = None
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
            for segment in group.segments:
                cols = int(segment.columns)
                base_tokens = int(segment.token_columns())
                if cols <= 0:
                    pos_cursor += base_tokens
                    continue
                mode = segment.mode
                connector = getattr(segment, "connector", None)
                if connector == "#":
                    kv_chain = []
                start = cursor
                end = cursor + cols
                extra_metric_templates = list(getattr(segment, "extra_metrics", ()))
                coord_metric_templates = [
                    name for name in extra_metric_templates if _metric_template_has_coords(name)
                ]
                plain_metric_templates = [
                    name for name in extra_metric_templates if not _metric_template_has_coords(name)
                ]
                sane_depth = max(1, int(getattr(segment, "sane_z", 1) or 1))
                sane_passes = max(1, int(getattr(segment, "sane_x", 1) or 1))
                restrict_active_zone = bool(getattr(segment, "sane_z_strict", False))
                use_sane_decode = mode == "decode" and sane_depth > 1
                if use_sane_decode:
                    if getattr(segment, "loss_input_stream", False) or getattr(segment, "loss_output_stream", False):
                        raise ValueError("SANE decode segments do not support b/B modifiers yet")
                    if base_tokens <= 0:
                        pos_cursor += base_tokens
                        cursor += cols
                        continue
                    sane_first_columns = None
                    plan = _sane_column_plan(base_tokens, sane_depth)
                    if len(plan) != cols:
                        raise ValueError("SANE decode plan does not match allocated columns")
                    base_start = pos_cursor
                    full_tokens = torch.cat([xb_base[:, :1], yb_extended], dim=1)
                    full_embeddings = model.core.expand_to_even(
                        model.core.tok_emb(full_tokens)
                    )
                    column_local = torch.tensor(
                        [idx for idx, _ in plan], device=device, dtype=torch.long
                    )
                    column_z = torch.tensor(
                        [z for _, z in plan], device=device, dtype=torch.long
                    )
                    sane_first_columns = (column_z == 0).nonzero(as_tuple=False).squeeze(-1)
                    sane_group_ids = column_local
                    sane_z_indices = column_z
                    column_base_offsets = column_local + base_start
                    column_offsets = column_base_offsets + column_z
                    next_offsets = column_offsets + 1
                    seq_len = full_tokens.size(1)
                    if torch.any(column_offsets >= seq_len) or torch.any(next_offsets >= seq_len):
                        raise ValueError("SANE decode segment requires unavailable tokens")
                    embedding_index = column_offsets.view(1, -1, 1).expand(
                        row_count, -1, full_embeddings.size(-1)
                    )
                    column_embeddings = torch.gather(full_embeddings, 1, embedding_index)

                    def _gather_token_ids(offsets: torch.Tensor) -> torch.Tensor:
                        index = offsets.view(1, -1).expand(row_count, -1)
                        return torch.gather(full_tokens, 1, index)

                    self_token_ids = _gather_token_ids(column_offsets)
                    next_token_ids = _gather_token_ids(next_offsets)
                    state_tensor = token_components.new_zeros(
                        row_count, cols, token_components.size(-1)
                    )
                    control_state = torch.full(
                        (cols,),
                        CONTROL_FIND_SELF,
                        dtype=torch.long,
                        device=device,
                    )
                    next_target_grid = torch.full(
                        (row_count, cols),
                        LOSS_IGNORE_INDEX,
                        dtype=xb_base.dtype,
                        device=xb_base.device,
                    )
                    self_target_grid = self_token_ids.clone()
                    kv_final: list[tuple[torch.Tensor, torch.Tensor]] | None = None
                    segment_loss: torch.Tensor | None = None
                    segment_tokens = 0
                    row_loss_sums_seg: torch.Tensor | None = None
                    row_token_counts_seg: torch.Tensor | None = None
                    for pass_idx in range(sane_passes):
                        chunk_base = state_tensor.clone()
                        newly_active = column_z == pass_idx
                        if torch.any(newly_active):
                            chunk_base[:, newly_active, :] = (
                                chunk_base[:, newly_active, :] + column_embeddings[:, newly_active, :]
                            )
                            control_state[newly_active] = CONTROL_PREDICT_NEXT
                            next_target_grid[:, newly_active] = next_token_ids[:, newly_active]
                        control_slice = (
                            control_state.view(1, -1).expand(row_count, -1).clone()
                        )
                        control_embed = model.core.expand_to_even(
                            model.core.control_emb(control_slice)
                        )
                        chunk_input = _compose_chunk_embeddings(
                            model.core.drop,
                            chunk_base,
                            control_slice=control_embed,
                        )
                        kv_sources = kv_chain if kv_chain else None
                        segment_positions = column_positions[start:end]
                        layer_repeat = max(1, int(getattr(segment, "layer_repeat", 1) or 1))
                        layer_top_only = bool(getattr(segment, "layer_top_only", False))
                        segment_disable_sane = row_disable_sane or getattr(segment, "disable_sane", False)
                        think_index = torch.full(
                            (cols,), pass_idx + 1, device=device, dtype=torch.long
                        )
                        think_count = torch.full(
                            (cols,), sane_passes, device=device, dtype=torch.long
                        )
                        active_mask_tensor = None
                        if restrict_active_zone:
                            active_mask_tensor = (column_z >= pass_idx).to(
                                device=device, dtype=torch.bool
                            )
                        chunk_output, grce_state, xctx_state, kv_out, layer_outputs = model.stack_sequence.forward(
                            chunk_input,
                            grce_in=grce_state,
                            xctx_in=xctx_state,
                            kv_cache_list_in=kv_sources,
                            mode=mode,
                            detach_internal_kv_cache=row_detach_kv_cache,
                            context_detach_span=detach_span_override,
                            context_detach_enabled=context_detach_override,
                            column_positions=segment_positions,
                            sane_first_columns=sane_first_columns,
                            sane_group_ids=sane_group_ids,
                            sane_z_indices=sane_z_indices,
                            sane_active_mask=active_mask_tensor,
                            think_step_index=think_index,
                            think_step_count=think_count,
                            layer_repeat=layer_repeat,
                            layer_top_only=layer_top_only,
                            think_last_only=bool(getattr(segment, "sane_x_last_only", False)),
                            capture_layer_outputs=False,
                            use_context=bool(getattr(segment, "context_enabled", True)),
                            disable_sane=segment_disable_sane,
                        )
                        kv_final = kv_out
                        state_tensor = model.core.sane_loop_norm(chunk_output)
                        head_features = model.core.ln_f(chunk_output)
                        next_logits = model.core.head(
                            model.core.output_features(head_features, use_next_stream=True)
                        )
                        self_logits = model.core.head(
                            model.core.output_features(head_features, use_next_stream=False)
                        )
                        next_targets = next_target_grid.clone()
                        self_targets = self_target_grid.clone()
                        predict_mask = next_targets != LOSS_IGNORE_INDEX
                        self_mask = self_targets != LOSS_IGNORE_INDEX
                        pass_loss: torch.Tensor | None = None
                        pass_tokens = 0
                        row_loss_combined: torch.Tensor | None = None
                        row_tokens_combined: torch.Tensor | None = None
                        if torch.any(predict_mask):
                            loss_val, tok_count, row_loss, row_tokens = loss_sum_token_count_with_rows(
                                next_logits,
                                next_targets,
                                last_only=False,
                            )
                            if tok_count > 0:
                                pass_loss = loss_val
                                pass_tokens += tok_count
                                row_loss_combined = row_loss
                                row_tokens_combined = row_tokens
                        if torch.any(self_mask):
                            loss_val, tok_count, row_loss, row_tokens = loss_sum_token_count_with_rows(
                                self_logits,
                                self_targets,
                                last_only=False,
                            )
                            if tok_count > 0:
                                pass_loss = loss_val if pass_loss is None else pass_loss + loss_val
                                pass_tokens += tok_count
                                if row_loss is not None:
                                    if row_loss_combined is None:
                                        row_loss_combined = row_loss
                                        row_tokens_combined = row_tokens
                                    else:
                                        row_loss_combined = row_loss_combined + row_loss
                                        row_tokens_combined = row_tokens_combined + row_tokens
                        if pass_loss is None or pass_tokens <= 0:
                            continue
                        segment_loss = pass_loss if segment_loss is None else segment_loss + pass_loss
                        segment_tokens += pass_tokens
                        metric_key = segment.metric_mode or mode
                        loss_value = float(pass_loss.detach().item())
                        if (
                            collect_mode_metrics
                            and not getattr(segment, "hide_typed_metrics", False)
                            and not getattr(segment, "suppress_default_metric", False)
                        ):
                            _accumulate_metric(
                                mode_loss_sums,
                                mode_token_counts,
                                metric_key,
                                loss_value,
                                pass_tokens,
                            )
                        for tag in plain_metric_templates:
                            _accumulate_metric(
                                mode_loss_sums,
                                mode_token_counts,
                                tag,
                                loss_value,
                                pass_tokens,
                            )
                        if coord_metric_templates:
                            per_column_losses = chunk_output.new_zeros(cols)
                            per_column_tokens = chunk_output.new_zeros(cols)
                            if torch.any(predict_mask):
                                loss_vec, token_vec = _column_loss_stats(next_logits, next_targets)
                                per_column_losses += loss_vec
                                per_column_tokens += token_vec
                            if torch.any(self_mask):
                                loss_vec, token_vec = _column_loss_stats(self_logits, self_targets)
                                per_column_losses += loss_vec
                                per_column_tokens += token_vec
                            loss_list = per_column_losses.detach().cpu().tolist()
                            token_list = [int(val) for val in per_column_tokens.detach().cpu().tolist()]
                            column_z_list = [int(val) for val in column_z.detach().cpu().tolist()]
                            for col_idx, tok_count in enumerate(token_list):
                                if tok_count <= 0:
                                    continue
                                per_loss = float(loss_list[col_idx])
                                z_value = column_z_list[col_idx]
                                for template in coord_metric_templates:
                                    metric_name = _format_metric_template(
                                        template,
                                        pass_idx,
                                        z_value,
                                    )
                                    _accumulate_metric(
                                        mode_loss_sums,
                                        mode_token_counts,
                                        metric_name,
                                        per_loss,
                                        tok_count,
                                    )
                        if row_loss_combined is not None and row_tokens_combined is not None:
                            loss_values = row_loss_combined.detach().cpu().tolist()
                            token_values = row_tokens_combined.detach().cpu().tolist()
                            for idx, entry in enumerate(row_entries):
                                if idx >= len(loss_values):
                                    break
                                entry["loss_sum"] = entry.get("loss_sum", 0.0) + float(loss_values[idx])
                                entry["token_count"] = (
                                    int(entry.get("token_count", 0)) + int(token_values[idx])
                                )
                    if segment_loss is not None:
                        total_loss_sum = segment_loss if total_loss_sum is None else total_loss_sum + segment_loss
                    total_tokens += segment_tokens
                    kv_chain.append(kv_final)
                    cursor += cols
                    pos_cursor += base_tokens
                    continue
                if mode == "reverse":
                    control_ids[:, start:end] = CONTROL_PREDICT_PREV
                elif mode in {"decode", "forward", "noattn"}:
                    control_ids[:, start:end] = CONTROL_PREDICT_NEXT
                elif mode == "encode" and end > start:
                    control_ids[:, start:end] = CONTROL_NONE
                    control_ids[:, end - 1 : end] = CONTROL_PREDICT_NEXT
                control_slice = control_ids[:, start:end].clone()
                control_embed = model.core.expand_to_even(
                    model.core.control_emb(control_slice)
                )
                if mode == "reverse":
                    token_source = future_token_components
                    chunk_target = xb[:, start:end]
                else:
                    token_source = token_components
                    chunk_target = yb[:, start:end]
                token_slice = token_source[:, start:end, :]
                drop_mask_tensor: torch.Tensor | None = None
                if mode == "encode":
                    drop_count = getattr(segment, "drop_count", 0)
                    if drop_count > 0:
                        drop_positions = _sample_drop_positions(cols, drop_count, rng)
                        drop_mask_tensor = torch.zeros(cols, dtype=torch.bool, device=token_slice.device)
                        drop_mask_tensor[drop_positions] = True
                        token_slice = token_slice.clone()
                        token_slice[:, drop_mask_tensor, ::2] = 0
                        control_slice = control_slice.clone()
                        control_slice[:, drop_mask_tensor] = CONTROL_FIND_SELF
                        control_ids[:, start:end][:, drop_mask_tensor] = CONTROL_FIND_SELF
                        control_embed = model.core.expand_to_even(
                            model.core.control_emb(control_slice)
                        )
                think_index, think_count, think_mask = _segment_think_metadata(
                    segment,
                    cols,
                    token_slice.device,
                )
                if think_index is not None:
                    keep = (think_index == 1).to(token_slice.dtype).view(1, -1, 1)
                    token_slice = token_slice * keep
                if think_mask is not None:
                    chunk_target = chunk_target.clone()
                    chunk_target[:, think_mask] = LOSS_IGNORE_INDEX
                think_slice = _segment_think_slice(
                    model.core,
                    segment,
                    row_count,
                    token_slice.device,
                )
                loop_slice = _segment_loop_slice(
                    model.core,
                    segment,
                    row_count,
                    token_slice.size(1),
                    token_slice.device,
                )
                chunk_input = _compose_chunk_embeddings(
                    model.core.drop,
                    token_slice,
                    control_slice=control_embed,
                    think_slice=think_slice,
                    loop_slice=loop_slice,
                )
                if mode == "reverse":
                    chunk_input = model.core.swap_self_next_streams(chunk_input)
                kv_sources = kv_chain if kv_chain else None
                prev_grce_state = grce_state
                prev_xctx_state = xctx_state
                segment_positions = column_positions[start:end]
                layer_repeat = max(1, int(getattr(segment, "layer_repeat", 1) or 1))
                layer_top_only = bool(getattr(segment, "layer_top_only", False))
                think_last_only = bool(getattr(segment, "think_last_only", False))
                capture_layers = bool(
                    getattr(segment, "loss_input_stream", False)
                    or getattr(segment, "loss_output_stream", False)
                )
                segment_disable_sane = row_disable_sane or getattr(segment, "disable_sane", False)
                extra_metric_templates = list(getattr(segment, "extra_metrics", ()))
                coord_metric_templates = [
                    name for name in extra_metric_templates if _metric_template_has_coords(name)
                ]
                plain_metric_templates = [
                    name for name in extra_metric_templates if not _metric_template_has_coords(name)
                ]
                chunk_output, grce_state, xctx_state, kv_out, layer_outputs = model.stack_sequence.forward(
                    chunk_input,
                    grce_in=grce_state,
                    xctx_in=xctx_state,
                    kv_cache_list_in=kv_sources,
                    mode=mode,
                    detach_internal_kv_cache=row_detach_kv_cache,
                    context_detach_span=detach_span_override,
                    context_detach_enabled=context_detach_override,
                    column_positions=segment_positions,
                    sane_first_columns=None,
                    sane_group_ids=None,
                    sane_z_indices=None,
                    think_step_index=think_index,
                    think_step_count=think_count,
                    layer_repeat=layer_repeat,
                    layer_top_only=layer_top_only,
                    think_last_only=think_last_only,
                    capture_layer_outputs=capture_layers,
                    use_context=bool(getattr(segment, "context_enabled", True)),
                    disable_sane=segment_disable_sane,
                )
                if not getattr(segment, "context_enabled", True):
                    if model.stack_sequence.grce is not None:
                        grce_state = model.stack_sequence.grce.initial_state(
                            row_count, chunk_output.device, chunk_output.dtype
                        )
                    else:
                        grce_state = None
                    if model.stack_sequence.xctx is not None:
                        xctx_state = model.stack_sequence.xctx.initial_state(
                            row_count, chunk_output.device, chunk_output.dtype
                        )
                    else:
                        xctx_state = None
                head_features = model.core.ln_f(chunk_output)
                use_next_stream = mode != "reverse"
                logits = model.core.head(
                    model.core.output_features(
                        head_features,
                        use_next_stream=use_next_stream,
                    )
                )
                eval_targets = chunk_target
                last_only = mode == "encode" and drop_mask_tensor is None
                if drop_mask_tensor is not None:
                    eval_targets = chunk_target.clone()
                    eval_targets[:, :] = LOSS_IGNORE_INDEX
                    eval_targets[:, drop_mask_tensor] = xb[:, start:end][:, drop_mask_tensor]
                    eval_targets[:, -1:] = chunk_target[:, -1:]
                use_standard_loss = not (
                    getattr(segment, "loss_input_stream", False)
                    or getattr(segment, "loss_output_stream", False)
                )
                if use_standard_loss:
                    (
                        loss_sum,
                        token_count,
                        row_loss_sums,
                        row_token_counts,
                    ) = loss_sum_token_count_with_rows(
                        logits,
                        eval_targets,
                        last_only=last_only,
                    )
                else:
                    loss_sum = None
                    token_count = _count_supervised_tokens(
                        eval_targets,
                        last_only=last_only,
                    )
                    row_loss_sums = None
                    row_token_counts = None
                bias_terms: list[torch.Tensor] = []
                if getattr(segment, "loss_output_stream", False):
                    if layer_outputs is None:
                        raise RuntimeError("Requested output-stream loss without captured layers")
                    output_loss = _layer_output_stream_loss(
                        model,
                        layer_outputs,
                        eval_targets,
                        last_only=last_only,
                        use_next_stream=use_next_stream,
                    )
                    if output_loss is not None:
                        bias_terms.append(output_loss)
                if getattr(segment, "loss_input_stream", False):
                    if layer_outputs is None:
                        raise RuntimeError("Requested input-stream loss without captured layers")
                    input_loss = _layer_input_stream_loss(
                        model,
                        layer_outputs,
                        token_slice,
                        use_next_stream=use_next_stream,
                    )
                    if input_loss is not None:
                        bias_terms.append(input_loss)
                if bias_terms:
                    bias_loss = sum(bias_terms) / len(bias_terms)
                    loss_sum = bias_loss
                if loss_sum is not None:
                    total_loss_sum = (
                        loss_sum if total_loss_sum is None else total_loss_sum + loss_sum
                    )
                if token_count > 0 and loss_sum is not None:
                    total_tokens += token_count
                    loss_value = float(loss_sum.detach().item())
                    metric_key = segment.metric_mode or mode
                    if (
                        collect_mode_metrics
                        and not getattr(segment, "hide_typed_metrics", False)
                        and not getattr(segment, "suppress_default_metric", False)
                    ):
                        _accumulate_metric(
                            mode_loss_sums,
                            mode_token_counts,
                            metric_key,
                            loss_value,
                            token_count,
                        )
                    for tag in plain_metric_templates:
                        _accumulate_metric(
                            mode_loss_sums,
                            mode_token_counts,
                            tag,
                            loss_value,
                            token_count,
                        )
                    if coord_metric_templates:
                        col_losses, col_tokens = _column_loss_stats(logits, eval_targets)
                        loss_list = col_losses.detach().cpu().tolist()
                        token_list = [int(val) for val in col_tokens.detach().cpu().tolist()]
                        for col_idx, tok_count in enumerate(token_list):
                            if tok_count <= 0:
                                continue
                            per_loss = float(loss_list[col_idx])
                            for template in coord_metric_templates:
                                metric_name = _format_metric_template(
                                    template,
                                    0,
                                    0,
                                )
                                _accumulate_metric(
                                    mode_loss_sums,
                                    mode_token_counts,
                                    metric_name,
                                    per_loss,
                                    tok_count,
                                )
                if row_loss_sums is not None and row_token_counts is not None:
                    loss_values = row_loss_sums.detach().cpu().tolist()
                    token_values = row_token_counts.detach().cpu().tolist()
                    for idx, entry in enumerate(row_entries):
                        if idx >= len(loss_values):
                            break
                        entry["loss_sum"] = entry.get("loss_sum", 0.0) + float(loss_values[idx])
                        entry["token_count"] = (
                            int(entry.get("token_count", 0)) + int(token_values[idx])
                        )
                kv_chain.append(kv_out)
                pos_cursor += base_tokens
                cursor += cols
            row_details.extend(row_entries)
    result = LayoutPassResult(
        total_loss_sum,
        total_tokens,
        base_token_total,
        dict(mode_loss_sums),
        dict(mode_token_counts),
    )
    return result, time.time() - start_time, row_details


def train_layout_batch(
    args: Args,
    model: GRCEGPT,
    dataset: TextDataset,
    layout: BatchLayout,
    device: torch.device,
    grad_hook: Callable[[int, int], None] | None = None,
    position_shift: int = 0,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, int, int, list[tuple[int, float, float, str]], dict[str, object]]:
    step_span = layout.total_token_span()
    if step_span <= 0:
        raise ValueError("Layout produced zero tokens for training step")
    window_rng = rng if rng is not None else random
    total_loss_sum: torch.Tensor | None = None
    total_tokens = 0
    total_base_tokens = 0
    micro_logs: list[tuple[int, float, float, str]] = []
    detail_entries: list[dict[str, object]] = []
    layout_text = layout.serialize()
    last_rows_text = ""
    collect_metrics = bool(layout.extra_metric_names)
    aggregated_mode_loss_sums: dict[str, float] = {}
    aggregated_mode_token_counts: dict[str, int] = {}
    for index, batch in enumerate(layout.micro_batches, start=1):
        micro_span = sum(row.token_span() for row in batch)
        if micro_span <= 0:
            micro_logs.append((index, 0.0, 0.0, layout.serialize_rows(batch)))
            continue
        rows_text = layout.serialize_rows(batch)
        last_rows_text = rows_text
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
                collect_mode_metrics=collect_metrics,
                position_shift=position_shift,
                rng=window_rng,
                row_serializer=layout.serialize_rows,
            )
            latest_metrics = getattr(args, "_latest_train_extra_metrics", {})
            if layout.extra_metric_names:
                for key in layout.extra_metric_names:
                    loss_sum = result.mode_loss_sums.get(key)
                    token_count = result.mode_token_counts.get(key, 0)
                    if loss_sum is not None and token_count > 0:
                        latest_metrics[key] = float(loss_sum / token_count)
            setattr(args, "_latest_train_extra_metrics", latest_metrics)
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
        total_base_tokens += result.base_tokens
        if collect_metrics:
            for key, value in result.mode_loss_sums.items():
                aggregated_mode_loss_sums[key] = aggregated_mode_loss_sums.get(key, 0.0) + value
            for key, value in result.mode_token_counts.items():
                aggregated_mode_token_counts[key] = aggregated_mode_token_counts.get(key, 0) + value
    if total_loss_sum is None:
        raise RuntimeError(
            f"Layout batch produced no tokens (last rows: {last_rows_text or '<none>'})"
        )
    meta_entry: dict[str, object] = {"rows": detail_entries}
    if collect_metrics:
        averages: dict[str, float] = {}
        for key, total in aggregated_mode_loss_sums.items():
            count = aggregated_mode_token_counts.get(key, 0)
            if count > 0:
                averages[key] = total / count
        meta_entry["extra_metrics"] = averages
    meta_entry["base_tokens"] = total_base_tokens
    return total_loss_sum, total_tokens, total_base_tokens, micro_logs, meta_entry


def evaluate_layout_batch(
    args: Args,
    model: GRCEGPT,
    dataset: TextDataset,
    split: str,
    layout: BatchLayout,
    device: torch.device,
    rng: random.Random | None = None,
) -> EvalBatchStats:
    step_span = layout.total_token_span()
    aggregate_loss_sums: dict[str, float] = {mode: 0.0 for mode in BATCH_MODES}
    aggregate_token_counts: dict[str, int] = {mode: 0 for mode in BATCH_MODES}
    total_loss_sum: torch.Tensor | None = None
    total_tokens = 0
    if step_span <= 0:
        base_keys = list(BATCH_MODES)
        extra_keys = list(layout.extra_metric_names)
        metrics: dict[str, float | None] = {mode: None for mode in base_keys + extra_keys}
        metrics["target"] = None
        loss_sums = {mode: 0.0 for mode in base_keys + extra_keys}
        loss_sums["target"] = 0.0
        token_counts = {mode: 0 for mode in base_keys + extra_keys}
        token_counts["target"] = 0
        return EvalBatchStats(metrics, loss_sums, token_counts)
    window_rng = rng if rng is not None else random
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
        for key, value in result.mode_loss_sums.items():
            aggregate_loss_sums[key] = aggregate_loss_sums.get(key, 0.0) + value
        for key, value in result.mode_token_counts.items():
            aggregate_token_counts[key] = aggregate_token_counts.get(key, 0) + value
    base_keys = list(BATCH_MODES)
    extra_keys = sorted(
        key for key in aggregate_loss_sums.keys() if key not in base_keys
    )
    all_keys = base_keys + extra_keys
    metrics: dict[str, float | None] = {mode: None for mode in all_keys}
    loss_sums = {mode: aggregate_loss_sums.get(mode, 0.0) for mode in all_keys}
    token_counts = {mode: aggregate_token_counts.get(mode, 0) for mode in all_keys}
    total_loss_value = float(total_loss_sum.item()) if total_loss_sum is not None else 0.0
    for mode in all_keys:
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


def _derive_cycle_seed(base_seed: int, cycle_index: int) -> int:
    """Mix the base RNG seed with the 1-based cycle index."""

    normalized_cycle = max(0, int(cycle_index) - 1)
    multiplier = 1_000_003
    return int(base_seed) + normalized_cycle * multiplier


def _derive_cycle_token_seed(
    base_seed: int,
    cycle_index: int,
    total_train_tokens: int,
    *,
    cycle_only: bool = False,
) -> int:
    seed = _derive_cycle_seed(base_seed, cycle_index)
    if cycle_only:
        return seed
    token_component = max(0, int(total_train_tokens))
    token_multiplier = 97_000_319
    return seed + token_component * token_multiplier


def _apply_global_rng_seed(seed: int) -> None:
    """Seed Python and Torch RNGs using the provided integer."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)


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


def _token_embeddings_with_offsets(
    model: GRCEGPT,
    token_batch: torch.Tensor,
    position_offsets: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, seq_len = token_batch.shape
    _ = model._position_ids(seq_len, batch_size, token_batch.device, position_offsets)
    return model.core.expand_to_even(model.core.tok_emb(token_batch))


def _sequence_embeddings(model: GRCEGPT, token_batch: torch.Tensor) -> torch.Tensor:
    """Project token IDs into dropout'd embeddings for stack sequence calls."""

    return _sequence_embeddings_with_offsets(model, token_batch, None)


def _sequence_embeddings_with_offsets(
    model: GRCEGPT,
    token_batch: torch.Tensor,
    position_offsets: torch.Tensor | None,
) -> torch.Tensor:
    tok = _token_embeddings_with_offsets(model, token_batch, position_offsets)
    return model.core.drop(tok)


def _compose_chunk_embeddings(
    dropout_layer: nn.Dropout,
    token_slice: torch.Tensor,
    *,
    control_slice: torch.Tensor | None = None,
    think_slice: torch.Tensor | None = None,
    loop_slice: torch.Tensor | None = None,
) -> torch.Tensor:
    base = token_slice
    if control_slice is not None:
        base = base + control_slice
    if think_slice is not None:
        base = base + think_slice
    if loop_slice is not None:
        base = base + loop_slice
    return dropout_layer(base)


def _expand_think_sequences(
    tokens: torch.Tensor,
    segments: Sequence[SegmentLayout],
) -> torch.Tensor:
    rows, base_cols = tokens.shape
    expected_positions = sum(segment.token_columns() for segment in segments)
    if base_cols != expected_positions:
        raise ValueError(
            f"Think expansion mismatch: got {base_cols} positions but layout expects {expected_positions}"
        )
    total_steps = sum(segment.columns for segment in segments)
    expanded = tokens.new_empty(rows, total_steps)
    base_cursor = 0
    step_cursor = 0
    for segment in segments:
        steps = int(segment.columns)
        if steps <= 0:
            continue
        factor = max(1, int(getattr(segment, "think_factor", 1) or 1))
        if factor <= 1:
            token_cols = int(segment.token_columns())
            if token_cols <= 0:
                expanded[:, step_cursor : step_cursor + steps] = 0
            elif token_cols == steps:
                expanded[:, step_cursor : step_cursor + steps] = tokens[
                    :, base_cursor : base_cursor + token_cols
                ]
            elif token_cols > steps:
                slice_end = base_cursor + steps
                expanded[:, step_cursor : step_cursor + steps] = tokens[:, base_cursor:slice_end]
            else:  # token_cols < steps
                slice_end = base_cursor + token_cols
                expanded[:, step_cursor : step_cursor + steps] = 0
                if token_cols > 0:
                    expanded[:, step_cursor : step_cursor + token_cols] = tokens[
                        :, base_cursor:slice_end
                    ]
            base_cursor += token_cols
            step_cursor += steps
            continue
        token_cols = steps // factor
        if token_cols <= 0:
            continue
        base_slice = tokens[:, base_cursor : base_cursor + token_cols]
        repeated = base_slice.repeat_interleave(factor, dim=1)
        expanded[:, step_cursor : step_cursor + steps] = repeated
        base_cursor += token_cols
        step_cursor += steps
    if step_cursor != total_steps or base_cursor != base_cols:
        raise ValueError("Think expansion bookkeeping error")
    return expanded


def _segment_think_slice(
    core: TransformerStackCore,
    segment: SegmentLayout,
    row_count: int,
    device: torch.device,
) -> torch.Tensor | None:
    factor = max(1, int(getattr(segment, "think_factor", 1) or 1))
    if factor <= 1 or segment.columns <= 0:
        return None
    tokens = segment.columns // factor
    if tokens <= 0:
        return None
    slot_sequence = core.think_emb.sequence(factor)
    template = slot_sequence.unsqueeze(0).repeat(tokens, 1, 1)
    chunk = template.view(tokens * factor, -1).to(device)
    expanded = core.expand_to_even(chunk)
    return expanded.unsqueeze(0).expand(row_count, -1, -1)


def _segment_loop_slice(
    core: TransformerStackCore,
    segment: SegmentLayout,
    row_count: int,
    cols: int,
    device: torch.device,
) -> torch.Tensor | None:
    if cols <= 0:
        return None
    repeat = max(1, int(getattr(segment, "layer_repeat", 1) or 1))
    if repeat <= 1:
        return None
    loop_vector = core.loop_embedding(repeat)
    if loop_vector is None:
        return None
    tiled = loop_vector.view(1, 1, -1)
    even = core.expand_to_even(tiled)
    return even.expand(row_count, cols, -1)


def _sample_drop_positions(cols: int, drop_count: int, rng: random.Random) -> list[int]:
    if drop_count <= 0:
        return []
    if cols <= 2:
        raise ValueError("Drop modifier requires at least three columns in the encoder segment")
    interior = list(range(1, cols - 1))
    if not interior:
        raise ValueError("Drop modifier requires interior columns to target")
    max_drop = (len(interior) + 1) // 2
    if drop_count > max_drop:
        raise ValueError(
            f"Drop modifier requests {drop_count} positions but at most {max_drop} are available"
        )
    available = interior[:]
    selected: list[int] = []
    while len(selected) < drop_count:
        candidates = [pos for pos in available if all(abs(pos - chosen) > 1 for chosen in selected)]
        if not candidates:
            raise ValueError("Unable to place non-consecutive drop positions within the segment")
        pos = rng.choice(candidates)
        selected.append(pos)
        available = [p for p in available if abs(p - pos) > 1]
    return sorted(selected)


def _segment_think_metadata(
    segment: SegmentLayout,
    cols: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    factor = max(1, int(getattr(segment, "think_factor", 1) or 1))
    if factor <= 1 or cols <= 0:
        return None, None, None
    idx = torch.arange(cols, device=device, dtype=torch.long)
    step_count = torch.full((cols,), factor, device=device, dtype=torch.long)
    step_index = (idx % factor) + 1
    mask = step_index != step_count
    return step_index, step_count, mask


def _layer_output_stream_loss(
    model: "GRCEGPT",
    layer_outputs: Sequence[torch.Tensor],
    targets: torch.Tensor,
    *,
    last_only: bool,
    use_next_stream: bool = True,
) -> torch.Tensor | None:
    total_loss: torch.Tensor | None = None
    layer_count = 0
    for tensor in layer_outputs:
        logits = model.core.head(
            model.core.output_features(
                model.core.ln_f(tensor),
                use_next_stream=use_next_stream,
            )
        )
        loss_sum, token_count = loss_sum_and_token_count(logits, targets, last_only=last_only)
        if token_count <= 0:
            continue
        total_loss = loss_sum if total_loss is None else total_loss + loss_sum
        layer_count += 1
    if total_loss is None or layer_count == 0:
        return None
    return total_loss / layer_count


def _layer_input_stream_loss(
    model: "GRCEGPT",
    layer_outputs: Sequence[torch.Tensor],
    token_slice: torch.Tensor,
    *,
    use_next_stream: bool = True,
) -> torch.Tensor | None:
    if not layer_outputs:
        return None
    target_ids = token_slice.argmax(dim=-1)
    accumulated: torch.Tensor | None = None
    layer_count = 0
    for tensor in layer_outputs:
        logits = model.core.head(
            model.core.output_features(
                model.core.ln_f(tensor),
                use_next_stream=use_next_stream,
            )
        )
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="mean",
        )
        accumulated = per_token if accumulated is None else accumulated + per_token
        layer_count += 1
    if accumulated is None or layer_count == 0:
        return None
    return accumulated / layer_count


def _count_supervised_tokens(targets: torch.Tensor, *, last_only: bool) -> int:
    if last_only:
        targets = targets[:, -1:]
    mask = targets != LOSS_IGNORE_INDEX
    return int(mask.sum().item())


def _column_loss_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.numel() == 0:
        shape = (targets.size(1),)
        return (
            logits.new_zeros(shape),
            logits.new_zeros(shape),
        )
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        ignore_index=LOSS_IGNORE_INDEX,
    ).view(targets.size(0), targets.size(1))
    mask = (targets != LOSS_IGNORE_INDEX).to(per_token.dtype)
    loss_per_col = (per_token * mask).sum(dim=0)
    token_per_col = mask.sum(dim=0)
    return loss_per_col, token_per_col


def _accumulate_metric(
    sums: dict[str, float],
    counts: dict[str, int],
    name: str,
    loss_value: float,
    token_count: int,
) -> None:
    if token_count <= 0:
        return
    if name not in sums:
        sums[name] = 0.0
        counts[name] = 0
    sums[name] += loss_value
    counts[name] += token_count


def _kv_column_length(kv_pairs: Sequence[tuple[torch.Tensor, torch.Tensor] | None]) -> int:
    for pair in kv_pairs:
        if pair is None:
            continue
        return pair[0].size(1)
    return 0


def _kv_history_prune_last(
    history: list[list[tuple[torch.Tensor, torch.Tensor] | None]],
    drop_columns: int,
) -> None:
    if drop_columns <= 0:
        return
    if len(history) < drop_columns + 1:
        return
    del history[-(drop_columns + 1):-1]


def _kv_storage_prune_lengths(lengths: list[int], drop_columns: int) -> int:
    if drop_columns <= 0:
        return 0
    if len(lengths) < drop_columns + 1:
        return 0
    start = len(lengths) - (drop_columns + 1)
    end = len(lengths) - 1
    removed = lengths[start:end]
    del lengths[start:end]
    return sum(removed)


def _segment_column_positions(
    segments: Sequence[SegmentLayout], device: torch.device
) -> torch.Tensor:
    total_cols = sum(max(0, int(seg.columns)) for seg in segments)
    positions = torch.zeros(total_cols, dtype=torch.long, device=device)
    cursor = 0
    position_cursor = 0
    for segment in segments:
        cols = int(segment.columns)
        if cols <= 0:
            continue
        factor = max(1, int(getattr(segment, "think_factor", 1) or 1))
        if factor <= 1:
            local = torch.arange(cols, dtype=torch.long, device=device)
            positions[cursor : cursor + cols] = local + position_cursor
            position_cursor += cols
        else:
            token_cols = cols // factor
            base = torch.arange(max(1, token_cols), dtype=torch.long, device=device)
            repeated = base.repeat_interleave(factor)
            if repeated.numel() > cols:
                repeated = repeated[:cols]
            elif repeated.numel() < cols:
                repeated = torch.nn.functional.pad(
                    repeated,
                    (0, cols - repeated.numel()),
                    mode="replicate",
                )
            positions[cursor : cursor + cols] = repeated + position_cursor
            position_cursor += token_cols
        cursor += cols
    return positions


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
    if hasattr(global_runtime_args, "checkpoint_dirty"):
        global_runtime_args.checkpoint_dirty = True


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
    block_size: int,
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
    train_step_index = start_step
    history_updates: List[Dict[str, float]] = []
    printed_header = False
    eval_interval = max(1, int(eval_interval))
    loop_timer = Timer().start()
    eval_timer = Timer()
    train_tokens_used = 0
    base_total_train_tokens = int(total_train_tokens_start)
    run_total_steps = max(1, args.steps * args.cycles)
    current_lr = args.lr_base

    setattr(args, "_latest_train_extra_metrics", {})

    header_probe = BatchLayout(
        args.layout,
        batch_size=batch_size,
        block_size=block_size,
        rng=random.Random(0),
    )
    seen_extra_metric_keys = getattr(args, "_seen_extra_metric_keys", None)
    if seen_extra_metric_keys is None:
        seen_extra_metric_keys = set()
        setattr(args, "_seen_extra_metric_keys", seen_extra_metric_keys)
    seen_extra_metric_keys.update(header_probe.extra_metric_names)
    header_extra_metric_keys = sorted(seen_extra_metric_keys)
    long_loss_header = " ".join(
        [""] + [f"{': ' if key in ROW_METRIC_LOG_GROUP else ''}{key}" for key in ROW_METRIC_LOG_KEYS]
    )

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
    if header_extra_metric_keys:
        header_columns.append(
            (
                " ".join(header_extra_metric_keys),
                Colors.WHITE,
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
            layout = BatchLayout(args.layout, batch_size=batch_size, block_size=block_size)
        layout_serialized = layout.serialize()
        layout_span = layout.total_token_span()
        _log_layout_warnings(args, layout)
        current_step_index = train_step_index + 1
        step_wall_start = time.time()
        step_base_tokens = 0
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
                max(0, train_step_index),
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
            if args.block_size < args.n_pos:
                headroom = max(0, args.n_pos - args.block_size)
                if headroom > 0:
                    position_shift = random.randint(0, headroom)
            total_loss_sum, total_tokens, base_tokens, micro_logs, window_detail = train_layout_batch(
                args,
                model,
                dataset,
                layout,
                device,
                grad_hook=_record_micro_grad if need_grad_tracking else None,
                position_shift=position_shift,
            )
            live_metrics = window_detail.get("extra_metrics")
            if isinstance(live_metrics, dict):
                previous_metrics = getattr(args, "_latest_train_extra_metrics", {})
                merged = dict(previous_metrics)
                merged.update(live_metrics)
                setattr(args, "_latest_train_extra_metrics", merged)
            step_base_tokens = base_tokens
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
            train_tokens_used += step_base_tokens
        except torch.OutOfMemoryError as oom_err:
            oom_retries += 1
            line_parts: List[str] = []
            if show_time:
                timestamp = time.strftime("%H:%M", time.localtime())
                line_parts.append(color_text(timestamp, Colors.BLUE))
            line_parts.append(color_text(f"{train_step_index}", Colors.CYAN))
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
                candidate = BatchLayout(args.layout, batch_size=batch_size, block_size=block_size)
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
        train_step_index += 1

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
            use_decode_mode=args.generate_with_decode,
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

        def format_metric(dataset_split: str, key: str, *, sep_override: str | None = None) -> str:
            value = eval_metrics[dataset_split].metrics.get(key)
            sep = sep_override if sep_override is not None else (
                ": " if key in ROW_METRIC_LOG_GROUP else ""
            )
            if value is None:
                return f"{sep}****"
            return f"{sep}{value:.2f}"

        def metric_value_text(dataset_split: str, key: str) -> str:
            value = eval_metrics[dataset_split].metrics.get(key)
            return "****" if value is None else f"{value:.2f}"

        detail_keys = ROW_METRIC_LOG_KEYS

        def format_train_line() -> str:
            base = format_metric("train", "target", sep_override="")
            if show_train_loss_details:
                diag = " ".join(format_metric("train", key) for key in detail_keys)
                if diag.strip():
                    base = f"{base} {diag}"
            return base

        def format_test_line() -> str:
            base = format_metric("test", "target", sep_override="")
            if show_test_loss_details:
                diag = " ".join(format_metric("test", key) for key in detail_keys)
                if diag.strip():
                    base = f"{base} {diag}"
            return base

        def format_extra_metrics_block(keys: Sequence[str]) -> str:
            if not keys:
                return ""
            parts = []
            live_train_metrics = getattr(args, "_latest_train_extra_metrics", {})
            for key in keys:
                live_value = live_train_metrics.get(key)
                if live_value is not None:
                    parts.append(f"{live_value:.2f}")
                else:
                    parts.append(metric_value_text("test", key))
            return " ".join(parts)

        newly_observed_extra_metrics = set(layout.extra_metric_names)
        newly_observed_extra_metrics.update(
            key
            for key in eval_metrics["train"].metrics.keys()
            if key not in ROW_METRIC_LOG_KEYS and key != "target"
        )
        newly_observed_extra_metrics.update(
            key
            for key in eval_metrics["test"].metrics.keys()
            if key not in ROW_METRIC_LOG_KEYS and key != "target"
        )
        seen_extra_metric_keys.update(newly_observed_extra_metrics)
        active_extra_metrics = sorted(seen_extra_metric_keys)

        train_values = format_train_line()
        test_values = format_test_line()
        extra_metrics_text = format_extra_metrics_block(active_extra_metrics)
        line_parts: List[str] = []
        if show_time:
            timestamp = time.strftime("%H:%M", time.localtime())
            line_parts.append(color_text(timestamp, Colors.BLUE))
        line_parts.append(color_text(f"{train_step_index}", Colors.CYAN))
        line_parts.append(color_text(train_values, Colors.MAGENTA))
        line_parts.append(color_text(test_values, Colors.GREEN))
        if extra_metrics_text:
            line_parts.append(color_text(extra_metrics_text, Colors.WHITE))
        line_parts.append(color_text(f"{current_lr:.2e}", Colors.YELLOW))
        line = " | ".join(line_parts) + " | " + sample_render
        print(line)

        eval_now = time.time()
        cycle_wall_elapsed = max(0.0, eval_now - cycle_wall_start)
        total_wall_seconds = base_wall_seconds + cycle_wall_elapsed

        current_total_train_tokens = base_total_train_tokens + train_tokens_used
        record = {
            "step": train_step_index,
            "train_loss": float(eval_metrics["train"].metrics.get("target", 0.0) or 0.0),
            "test_loss": float(eval_metrics["test"].metrics.get("target", 0.0) or 0.0),
            "train_wall_seconds": float(total_wall_seconds),
            "unix_time": float(eval_now),
            "train_tokens": step_base_tokens,
            "total_train_tokens": int(current_total_train_tokens),
        }
        record["corpus"] = args.corpus
        record["batch_layout"] = layout_serialized
        record["learning_rate"] = current_lr
        metric_keys = list(ROW_METRIC_HIST_KEYS)
        extra_history_keys = sorted(
            key
            for key in eval_metrics["train"].metrics.keys()
            if key not in metric_keys and key != "target"
        )
        metric_keys.extend(extra_history_keys)
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

    return train_step_index, history_updates, loop_timer.stop(), eval_timer, train_tokens_used


def run_profile_mode(
    args: Args,
    dataset: TextDataset,
    model: GRCEGPT,
    optimizer: torch.optim.Optimizer,
    *,
    block_size: int,
    batch_size: int,
    device: torch.device,
    n_pos: int,
    ) -> None:
    """Warm up once, profile a second training step, and report CUDA stats."""

    try:
        from torch.profiler import ProfilerActivity, profile, record_function
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "torch.profiler is unavailable; upgrade to PyTorch 1.8+ to use 'profile'."
        ) from exc

    profile_layout = BatchLayout(args.layout, batch_size=batch_size, block_size=block_size)
    _log_layout_warnings(args, profile_layout)

    def train_step(tag: str, layout: BatchLayout) -> float:
        model.train()
        position_shift = 0
        if block_size < n_pos:
            headroom = max(0, n_pos - block_size)
            if headroom > 0:
                position_shift = random.randint(0, headroom)
        total_loss_sum, total_tokens, _, _, _ = train_layout_batch(
            args,
            model,
            dataset,
            layout,
            device,
            position_shift=position_shift,
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
    use_decode_mode: bool = False,
) -> tuple[torch.Tensor, int]:
    """Autoregressively sample tokens for CLI reports and prompt tests."""

    model.eval()
    idx = idx.clone()
    prompt_len = idx.size(1)
    enforce_first_token_guard = bool(first_token_blocklist)
    blocklist = list(first_token_blocklist or [])
    for _ in range(steps):
        idx_cond = idx[:, -model.config.n_pos :]
        logits, _, _ = model.forward_autoreg(
            idx_cond,
            mode="decode" if use_decode_mode else "forward",
        )
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
    *,
    use_decode_mode: bool = False,
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
            use_decode_mode=use_decode_mode,
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
    source_ids: torch.Tensor
    attention_maps: dict[int, dict[int, torch.Tensor]] | None = None
    block_attentions: list["BlockAttention"] | None = None
    sampled_rows: int = 1


@dataclass
class BlockAttention:
    mode: str
    tokens: list[int]
    target_token: int
    layer_weights: dict[int, torch.Tensor]


@dataclass
class EvalSummary:
    loss_sums: dict[str, float]
    token_counts: dict[str, int]
    column_loss_sums: list[float]
    column_token_counts: list[int]
    source_label: str


def _evaluate_row_block(
    args: Args,
    model: GRCEGPT,
    row: BlockLayout,
    base_inputs: torch.Tensor,
    base_targets: torch.Tensor,
    *,
    capture_columns: set[int] | None = None,
    drop_rng: random.Random | None = None,
) -> RowEvalResult:
    rows, available_tokens = base_inputs.shape
    cols_total = row.total_columns()
    pos_total = row.total_positions()
    if cols_total <= 0 or pos_total <= 0:
        raise ValueError("Row block does not contain any tokens to evaluate")
    if available_tokens < pos_total:
        raise ValueError(
            f"Row requires {pos_total} base tokens but only {available_tokens} were provided"
        )
    base_inputs = base_inputs[:, :pos_total]
    base_targets = base_targets[:, :pos_total]
    expanded_inputs = _expand_think_sequences(base_inputs, row.segments)
    expanded_targets = _expand_think_sequences(base_targets, row.segments)
    row_count = expanded_inputs.size(0)
    token_components = _token_embeddings_with_offsets(
        model,
        expanded_inputs,
        None,
    )
    future_token_components = _token_embeddings_with_offsets(
        model,
        expanded_targets,
        None,
    )
    column_positions = _segment_column_positions(row.segments, token_components.device)
    if row.modifiers and getattr(row.modifiers, "halt_rope", False):
        column_positions = torch.zeros_like(column_positions)
    device = token_components.device
    column_modes: list[str] = [""] * cols_total
    supervision_mask = torch.ones(cols_total, dtype=torch.bool, device=device)
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
    row_disable_sane = bool(modifiers and getattr(modifiers, "disable_sane", False))
    control_ids = torch.zeros((row_count, cols_total), dtype=torch.long, device=device)
    drop_rng = drop_rng or random.Random(getattr(args, "rng_seed", 0) or 0)
    target_ids = torch.zeros_like(expanded_targets)
    for segment in row.segments:
        segment_start = cursor
        cols = int(segment.columns)
        if cols <= 0:
            continue
        if cursor + cols > cols_total:
            raise ValueError("Layout segment exceeds available token columns during evaluation")
        connector = getattr(segment, "connector", None)
        if connector == "#":
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
        control_slice = control_ids[:, start:end].clone()
        drop_mask_tensor: torch.Tensor | None = None
        if segment.mode == "encode" and getattr(segment, "drop_count", 0) > 0:
            drop_positions = _sample_drop_positions(cols, segment.drop_count, drop_rng)
            drop_mask_tensor = torch.zeros(cols, dtype=torch.bool, device=device)
            drop_mask_tensor[drop_positions] = True
            token_slice = token_slice.clone()
            token_slice[:, drop_mask_tensor, ::2] = 0
            control_slice = control_slice.clone()
            control_slice[:, drop_mask_tensor] = CONTROL_FIND_SELF
            control_ids[:, start:end][:, drop_mask_tensor] = CONTROL_FIND_SELF
        control_embed = model.core.expand_to_even(model.core.control_emb(control_slice))
        if segment.mode == "reverse":
            token_source = future_token_components
            chunk_target = expanded_inputs[:, start:end]
        else:
            token_source = token_components
            chunk_target = expanded_targets[:, start:end]
        token_slice = token_source[:, start:end, :]
        think_index, think_count, think_mask = _segment_think_metadata(
            segment,
            cols,
            token_slice.device,
        )
        if think_index is not None:
            keep = (think_index == 1).to(token_slice.dtype).view(1, -1, 1)
            token_slice = token_slice * keep
        eval_targets = chunk_target
        if think_mask is not None:
            eval_targets = chunk_target.clone()
            eval_targets[:, think_mask] = LOSS_IGNORE_INDEX
        think_slice = _segment_think_slice(
            model.core,
            segment,
            row_count,
            token_slice.device,
        )
        loop_slice = _segment_loop_slice(
            model.core,
            segment,
            row_count,
            token_slice.size(1),
            token_slice.device,
        )
        chunk_input = _compose_chunk_embeddings(
            model.core.drop,
            token_slice,
            control_slice=control_embed,
            think_slice=think_slice,
            loop_slice=loop_slice,
        )
        if segment.mode == "reverse":
            chunk_input = model.core.swap_self_next_streams(chunk_input)
        chunk_capture = None
        if base_capture is not None:
            chunk_capture = base_capture.subset(cursor, cols)
        kv_sources = kv_chain if kv_chain else None
        prev_grce_state = grce_state
        prev_xctx_state = xctx_state
        segment_positions = column_positions[start:end]
        layer_repeat = max(1, int(getattr(segment, "layer_repeat", 1) or 1))
        layer_top_only = bool(getattr(segment, "layer_top_only", False))
        think_last_only = bool(getattr(segment, "think_last_only", False))
        capture_layers = bool(
            getattr(segment, "loss_input_stream", False)
            or getattr(segment, "loss_output_stream", False)
        )
        segment_disable_sane = row_disable_sane or getattr(segment, "disable_sane", False)
        extra_metric_templates = list(getattr(segment, "extra_metrics", ()))
        coord_metric_templates = [
            name for name in extra_metric_templates if _metric_template_has_coords(name)
        ]
        plain_metric_templates = [
            name for name in extra_metric_templates if not _metric_template_has_coords(name)
        ]
        chunk_output, grce_state, xctx_state, kv_out, layer_outputs = model.stack_sequence.forward(
            chunk_input,
            grce_in=grce_state,
            xctx_in=xctx_state,
            kv_cache_list_in=kv_sources,
            mode=segment.mode,
            detach_internal_kv_cache=row_detach_kv_cache,
            context_detach_span=detach_span_override,
            context_detach_enabled=context_detach_override,
            attention_capture=chunk_capture,
            column_positions=segment_positions,
            think_step_index=think_index,
            think_step_count=think_count,
            layer_repeat=layer_repeat,
            layer_top_only=layer_top_only,
            think_last_only=think_last_only,
            capture_layer_outputs=capture_layers,
            use_context=bool(segment.context_enabled),
            disable_sane=segment_disable_sane,
        )
        if not segment.context_enabled:
            batch_rows = chunk_output.size(0)
            if model.stack_sequence.grce is not None:
                grce_state = model.stack_sequence.grce.initial_state(
                    batch_rows, chunk_output.device, chunk_output.dtype
                )
            else:
                grce_state = None
            if model.stack_sequence.xctx is not None:
                xctx_state = model.stack_sequence.xctx.initial_state(
                    batch_rows, chunk_output.device, chunk_output.dtype
                )
            else:
                xctx_state = None
        head_features = model.core.ln_f(chunk_output)
        use_next_stream = segment.mode != "reverse"
        logits = model.core.head(
            model.core.output_features(head_features, use_next_stream=use_next_stream)
        )
        logits_buffer.append(logits)
        last_only = segment.mode == "encode" and drop_mask_tensor is None
        eval_targets = chunk_target
        if drop_mask_tensor is not None:
            eval_targets = chunk_target.clone()
            eval_targets[:, drop_mask_tensor] = expanded_inputs[:, start:end][:, drop_mask_tensor]
            ignore_mask = torch.ones(cols, dtype=torch.bool, device=device)
            ignore_mask[drop_mask_tensor] = False
            if cols > 0:
                ignore_mask[-1] = False
            eval_targets[:, ignore_mask] = LOSS_IGNORE_INDEX
        target_ids[:, start:end] = eval_targets
        use_standard_loss = not (
            getattr(segment, "loss_input_stream", False)
            or getattr(segment, "loss_output_stream", False)
        )
        if use_standard_loss:
            loss_sum, token_count = loss_sum_and_token_count(
                logits,
                eval_targets,
                last_only=last_only,
            )
        else:
            loss_sum = None
            token_count = _count_supervised_tokens(
                eval_targets,
                last_only=last_only,
            )
        extra_losses: list[torch.Tensor] = []
        if getattr(segment, "loss_output_stream", False):
            if layer_outputs is None:
                raise RuntimeError("Requested output-stream loss without captured layers")
            output_loss = _layer_output_stream_loss(
                model,
                layer_outputs,
                eval_targets,
                last_only=last_only,
                use_next_stream=use_next_stream,
            )
            if output_loss is not None:
                extra_losses.append(output_loss)
        if getattr(segment, "loss_input_stream", False):
            if layer_outputs is None:
                raise RuntimeError("Requested input-stream loss without captured layers")
            input_loss = _layer_input_stream_loss(
                model,
                layer_outputs,
                token_slice,
                use_next_stream=use_next_stream,
            )
            if input_loss is not None:
                extra_losses.append(input_loss)
        if extra_losses:
            bias_loss = sum(extra_losses) / len(extra_losses)
            loss_sum = bias_loss
        metric_key = segment.metric_mode or segment.mode
        column_modes[start:end] = [metric_key] * (end - start)
        if token_count > 0 and loss_sum is not None:
            loss_value = float(loss_sum.detach().item())
            if not getattr(segment, "hide_typed_metrics", False) and not segment.suppress_default_metric:
                if metric_key not in mode_loss_sums:
                    mode_loss_sums[metric_key] = 0.0
                    mode_token_counts[metric_key] = 0
                mode_loss_sums[metric_key] += loss_value
                mode_token_counts[metric_key] += token_count
            for tag in plain_metric_templates:
                if tag not in mode_loss_sums:
                    mode_loss_sums[tag] = 0.0
                    mode_token_counts[tag] = 0
                mode_loss_sums[tag] += loss_value
                mode_token_counts[tag] += token_count
            if coord_metric_templates:
                col_losses, col_tokens = _column_loss_stats(logits, eval_targets)
                loss_list = col_losses.detach().cpu().tolist()
                token_list = [int(val) for val in col_tokens.detach().cpu().tolist()]
                for col_idx, tok_count in enumerate(token_list):
                    if tok_count <= 0:
                        continue
                    per_loss = float(loss_list[col_idx])
                    for template in coord_metric_templates:
                        metric_name = _format_metric_template(
                            template,
                            0,
                            0,
                        )
                        if metric_name not in mode_loss_sums:
                            mode_loss_sums[metric_name] = 0.0
                            mode_token_counts[metric_name] = 0
                        mode_loss_sums[metric_name] += per_loss
                        mode_token_counts[metric_name] += tok_count
            total_loss += loss_value
            total_tokens += token_count
        kv_chain.append(kv_out)
        for local_idx in range(cols):
            idx = cursor + local_idx
            if segment.mode == "encode" and local_idx < cols - 1:
                if drop_mask_tensor is None or not bool(drop_mask_tensor[local_idx]):
                    supervision_mask[idx] = False
            elif think_mask is not None and bool(think_mask[local_idx]):
                supervision_mask[idx] = False
        if chunk_capture is not None and segment.mode in {"decode", "reverse", "encode"}:
            abs_col = chunk_capture.absolute_offset + (cols - 1)
            layer_weights: dict[int, torch.Tensor] = {}
            for layer_idx, store in attention_storage.items():
                weights = store.get(abs_col)
                if weights is not None:
                    layer_weights[layer_idx] = weights[:, -cols:].contiguous()
            if layer_weights:
                block_tokens = expanded_inputs[0, segment_start : segment_start + cols].tolist()
                target_tensor = target_ids[0, segment_start + cols - 1]
                target_token = int(target_tensor.item())
                block_attentions.append(
                    BlockAttention(
                        mode=metric_key,
                        tokens=block_tokens,
                        target_token=target_token,
                        layer_weights=layer_weights,
                    )
                )
        cursor += cols
        pos_cursor += base_tokens
    if cursor != cols_total:
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
        source_ids=expanded_inputs,
        attention_maps=attention_storage if capture_columns else None,
        block_attentions=block_attentions,
        sampled_rows=row_count,
    )


def _prepare_eval_tokens(
    args: Args,
    dataset: TextDataset,
    tokenizer: GPT2TokenizerWrapper,
    *,
    token_length: int,
    start_pos: int,
    custom_text: str | None,
    align_rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str, str]:
    if custom_text:
        provided = tokenizer.encode(custom_text)
        if provided.numel() < 2:
            raise ValueError("Custom text must produce at least two tokens for evaluation")
        if provided.numel() - 1 > args.n_pos:
            raise ValueError(
                "Custom text exceeds the configured --n-pos; shorten the text or increase --n-pos."
            )
        context_tokens = provided
        source_label = "custom text"
    else:
        span = token_length + 1
        if span <= 1:
            raise ValueError("--block-size must be >= 1 for evaluation")
        tokens = dataset._tokens_for_split("test")
        chunk, adjusted = dataset._aligned_window(
            tokens,
            start_pos,
            span,
            rng=align_rng,
            allow_resample=align_rng is not None,
        )
        context_tokens = chunk
        source_label = f"test split offset {adjusted}"
    if context_tokens.numel() < 2:
        raise ValueError("Not enough tokens collected for evaluation")
    inputs = context_tokens[:-1]
    targets = context_tokens[1:]
    if inputs.numel() < token_length:
        raise ValueError(
            f"Requested {token_length} evaluation tokens but only {inputs.numel()} available"
        )
    inputs = inputs[:token_length]
    targets = targets[:token_length]
    eval_block_size = inputs.numel()
    pretty_text = tokenizer.decode_pretty(args, context_tokens)
    return context_tokens, inputs, targets, eval_block_size, source_label, pretty_text


def _prepare_eval_batch_tokens(
    args: Args,
    dataset: TextDataset,
    tokenizer: GPT2TokenizerWrapper,
    *,
    token_length: int,
    start_positions: Sequence[int],
    align_rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    if not start_positions:
        raise ValueError("Random evaluation batch requires at least one start position")
    input_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    labels: list[str] = []
    previews: list[str] = []
    for pos in start_positions:
        (
            _,
            inputs,
            targets,
            _,
            source_label,
            pretty_text,
        ) = _prepare_eval_tokens(
            args,
            dataset,
            tokenizer,
            token_length=token_length,
            start_pos=pos,
            custom_text=None,
            align_rng=align_rng,
        )
        input_chunks.append(inputs)
        target_chunks.append(targets)
        labels.append(source_label)
        if len(previews) < 3:
            previews.append(pretty_text)
    stacked_inputs = torch.stack(input_chunks)
    stacked_targets = torch.stack(target_chunks)
    source_label = ", ".join(labels)
    pretty_preview = "\n---\n".join(previews)
    return stacked_inputs, stacked_targets, source_label, pretty_preview


def run_test_slice(
    args: Args,
    dataset: TextDataset,
    tokenizer: GPT2TokenizerWrapper,
    model: GRCEGPT,
    block_size: int,
    start_pos: int,
    *,
    custom_text: str | None = None,
) -> None:
    """Run the layout on either a corpus slice or custom text and log per-token stats."""

    if block_size <= 0 and not custom_text:
        raise ValueError("--block-size must be positive for corpus-based test slices")

    model_device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    with torch.no_grad():
        layout = BatchLayout(args.layout, batch_size=args.batch_size, block_size=block_size)
        _log_layout_warnings(args, layout)
        max_positions = max((row.total_positions() for row in layout.rows), default=0)
        if max_positions <= 0:
            raise ValueError("Layout does not contain any token positions to evaluate")
        (
            context_tokens,
            inputs,
            targets,
            eval_token_length,
            source_label,
            pretty_text,
        ) = _prepare_eval_tokens(
            args,
            dataset,
            tokenizer,
            token_length=max_positions,
            start_pos=start_pos,
            custom_text=custom_text,
            align_rng=None,
        )

        print(color_text(f"Evaluating layout '{args.layout}' on {source_label}:", Colors.CYAN))
        print(pretty_text)
        print(color_text(
            f"Sequence tokens: {eval_token_length} inputs (context) + 1 target tail", Colors.CYAN
        ))

        xb_base = inputs.unsqueeze(0).to(model_device)
        yb_base = targets.unsqueeze(0).to(model_device)
        vocab_size = model.config.vocab_size

        eval_drop_rng = random.Random(getattr(args, "rng_seed", 0) or 0)
        for row_idx, row in enumerate(layout.rows, start=1):
            row_desc = _row_description(layout, row)
            print(color_text(f"Row block #{row_idx}: {row_desc}", Colors.YELLOW))

            capture_columns = None
            if getattr(args, "attn_map", False):
                row_steps = row.total_columns()
                if row_steps > 0:
                    capture_columns = {row_steps - 1}
            try:
                row_result = _evaluate_row_block(
                    args,
                    model,
                    row,
                    xb_base,
                    yb_base,
                    capture_columns=capture_columns,
                    drop_rng=eval_drop_rng,
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
            inputs_cpu = row_result.source_ids.squeeze(0).cpu().tolist()
            targets_cpu = row_result.target_ids.squeeze(0).cpu().tolist()
            modes_cpu = row_result.column_modes
            top_indices_cpu = top_indices.squeeze(0).cpu().tolist()
            top_probs_cpu = top_probs.squeeze(0).cpu().tolist()

            idx_width = 4
            token_width = max(
                len(_format_token_fragment(tokenizer, tok)) for tok in inputs_cpu + targets_cpu
            )
            pad = " " * 4
            seq_len = len(inputs_cpu)
            for col in range(seq_len):
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

def _metric_bucket_keys(metric_map: dict[str, float | None]) -> list[str]:
    extras = sorted(
        key for key in metric_map.keys() if key not in METRIC_BUCKET_ORDER
    )
    ordered: list[str] = []
    for key in METRIC_BUCKET_ORDER:
        if key in metric_map:
            ordered.append(key)
    ordered.extend(extras)
    return ordered


def _format_metric_map(metric_map: dict[str, float | None]) -> list[str]:
    lines: list[str] = []
    for key in _metric_bucket_keys(metric_map):
        value = metric_map.get(key)
        text = "****" if value is None else f"{value:.3f}"
        lines.append(f"  {key:>12}: {text}")
    if not lines:
        lines.append("  (no metrics recorded)")
    return lines


def _format_metric_inline(metric_map: dict[str, float | None]) -> str:
    parts: list[str] = []
    for key in _metric_bucket_keys(metric_map):
        value = metric_map.get(key)
        text = "****" if value is None else f"{value:.3f}"
        parts.append(f"{key}={text}")
    return ", ".join(parts) if parts else "(no metrics recorded)"


def _row_metric_values(row_result: RowEvalResult) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    metrics["target"] = (
        row_result.total_loss_sum / row_result.total_tokens
        if row_result.total_tokens > 0
        else None
    )
    for key, loss_sum in row_result.mode_loss_sums.items():
        count = row_result.mode_token_counts.get(key, 0)
        metrics[key] = loss_sum / count if count > 0 else None
    for key in METRIC_BUCKET_ORDER:
        metrics.setdefault(key, None)
    return metrics


def _overall_metric_values(
    loss_sums: dict[str, float],
    token_counts: dict[str, int],
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for key, total in loss_sums.items():
        count = token_counts.get(key, 0)
        metrics[key] = total / count if count > 0 else None
    for key in METRIC_BUCKET_ORDER:
        metrics.setdefault(key, None)
    return metrics


def _column_loss_values(
    row_result: RowEvalResult,
) -> tuple[list[float], list[bool]]:
    log_probs = torch.log_softmax(row_result.logits, dim=-1)
    targets = row_result.target_ids.unsqueeze(-1)
    gathered = torch.gather(log_probs, dim=-1, index=targets).squeeze(-1)
    if gathered.dim() == 2:
        reduced = gathered
    elif gathered.dim() == 1:
        reduced = gathered.unsqueeze(0)
    else:
        raise ValueError("Unexpected logits shape for column loss computation")
    losses = (-reduced).mean(dim=0).detach().cpu().tolist()
    mask = row_result.supervision_mask.detach().cpu().tolist()
    return losses, mask


def _column_summary_lines(
    loss_sums: Sequence[float],
    token_counts: Sequence[int],
) -> list[str]:
    values: list[str] = []
    for loss_sum, count in zip(loss_sums, token_counts):
        if count <= 0:
            continue
        avg = loss_sum / count
        values.append(f"{avg:.3f}")
    if not values:
        return ["  (no supervised columns)"]
    chunk_size = 20
    lines: list[str] = []
    for chunk_idx in range(0, len(values), chunk_size):
        chunk = values[chunk_idx : chunk_idx + chunk_size]
        prefix = "  [" if chunk_idx == 0 else "    "
        line = prefix
        if chunk_idx == 0:
            line += " "
        line += ", ".join(chunk)
        if chunk_idx + chunk_size < len(values):
            line += ","
        else:
            line += " ]"
        lines.append(line)
    return lines


def _aggregate_eval_summaries(
    summaries: Sequence[EvalSummary],
) -> tuple[dict[str, float], dict[str, int], list[float], list[int]]:
    total_loss: dict[str, float] = {}
    total_counts: dict[str, int] = {}
    max_columns = max((len(summary.column_loss_sums) for summary in summaries), default=0)
    column_loss = [0.0] * max_columns
    column_counts = [0] * max_columns
    for summary in summaries:
        for key, value in summary.loss_sums.items():
            total_loss[key] = total_loss.get(key, 0.0) + value
        for key, value in summary.token_counts.items():
            total_counts[key] = total_counts.get(key, 0) + value
        for idx in range(len(summary.column_loss_sums)):
            column_loss[idx] += summary.column_loss_sums[idx]
            column_counts[idx] += summary.column_token_counts[idx]
    return total_loss, total_counts, column_loss, column_counts


def _print_eval_summary(
    label: str,
    layout_text: str,
    summaries: Sequence[EvalSummary],
) -> None:
    if not summaries:
        return
    total_loss, total_counts, column_loss, column_counts = _aggregate_eval_summaries(summaries)
    metrics = _overall_metric_values(total_loss, total_counts)
    header = f"{label} layout '{layout_text}' across {len(summaries)} evaluation(s)"
    print(color_text(header, Colors.CYAN))
    offsets: list[str] = []
    for summary in summaries:
        match = re.search(r"offset (\d+)", summary.source_label)
        if match:
            offsets.append(match.group(1))
    if offsets:
        print(f"  offsets: {', '.join(offsets)}")
    else:
        sources = ", ".join(summary.source_label for summary in summaries)
        print(f"  sources: {sources}")
    print(color_text("Metric buckets:", Colors.CYAN))
    for line in _format_metric_map(metrics):
        print(line)
    print(color_text("Per-column losses:", Colors.CYAN))
    for line in _column_summary_lines(column_loss, column_counts):
        print(line)


def run_eval_layout(
    args: Args,
    dataset: TextDataset,
    tokenizer: GPT2TokenizerWrapper,
    model: GRCEGPT,
    block_size: int,
    start_pos: int,
    *,
    custom_text: str | None = None,
    verbose: bool | None = None,
    align_rng: random.Random | None = None,
    batch_inputs: torch.Tensor | None = None,
    batch_targets: torch.Tensor | None = None,
    batch_label: str | None = None,
    batch_pretty: str | None = None,
    use_sampled_weight: bool = False,
) -> EvalSummary:
    """Evaluate the layout on a deterministic slice and print per-row metrics."""

    if block_size <= 0 and not custom_text:
        raise ValueError("--block-size must be positive for corpus-based evaluation")

    model_device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    verbose_flag = bool(verbose if verbose is not None else getattr(args, "eval_verbose", False))

    with torch.no_grad():
        layout = BatchLayout(args.layout, batch_size=args.batch_size, block_size=block_size)
        _log_layout_warnings(args, layout)
        max_positions = max((row.total_positions() for row in layout.rows), default=0)
        if max_positions <= 0:
            raise ValueError("Layout does not contain any token positions to evaluate")
        if batch_inputs is not None and batch_targets is not None:
            inputs_tensor = batch_inputs
            targets_tensor = batch_targets
            source_label = batch_label or "random batch"
            pretty_text = batch_pretty or ""
            eval_block_size = inputs_tensor.size(-1)
        else:
            (
                context_tokens,
                inputs,
                targets,
                eval_block_size,
                source_label,
                pretty_text,
            ) = _prepare_eval_tokens(
                args,
                dataset,
                tokenizer,
                token_length=max_positions,
                start_pos=start_pos,
                custom_text=custom_text,
                align_rng=align_rng,
            )
            inputs_tensor = inputs.unsqueeze(0)
            targets_tensor = targets.unsqueeze(0)

        if verbose_flag and pretty_text:
            print(color_text(f"Evaluating layout '{args.layout}' on {source_label}:", Colors.CYAN))
            print(pretty_text)

        xb_base = inputs_tensor.to(model_device)
        yb_base = targets_tensor.to(model_device)

        overall_loss_sums = {mode: 0.0 for mode in BATCH_MODES}
        overall_token_counts = {mode: 0 for mode in BATCH_MODES}
        overall_loss_sums["target"] = 0.0
        overall_token_counts["target"] = 0
        max_columns = max((row.total_columns() for row in layout.rows), default=0)
        column_loss_sums = [0.0] * max_columns
        column_token_counts = [0] * max_columns

        eval_drop_rng = random.Random(getattr(args, "rng_seed", 0) or 0)
        for row_idx, row in enumerate(layout.rows, start=1):
            row_desc = _row_description(layout, row)
            try:
                row_result = _evaluate_row_block(
                    args,
                    model,
                    row,
                    xb_base,
                    yb_base,
                    drop_rng=eval_drop_rng,
                )
            except ValueError as exc:
                print(color_text(f"Row block #{row_idx}: {row_desc} -> error: {exc}", Colors.RED))
                continue

            row_metrics = _row_metric_values(row_result)
            if verbose_flag:
                inline_metrics = _format_metric_inline(row_metrics)
                print(
                    f"Row block #{row_idx}: {row_desc} (rows={row.rows}) -> {inline_metrics}"
                )

            sampled_rows = max(0, int(row_result.sampled_rows))
            weight = max(0, int(row.rows))
            if use_sampled_weight:
                weight = sampled_rows
            if weight > 0:
                overall_loss_sums["target"] += row_result.total_loss_sum * weight
                overall_token_counts["target"] += row_result.total_tokens * weight
                for mode, value in row_result.mode_loss_sums.items():
                    overall_loss_sums.setdefault(mode, 0.0)
                    overall_token_counts.setdefault(mode, 0)
                    overall_loss_sums[mode] += value * weight
                    overall_token_counts[mode] += row_result.mode_token_counts.get(mode, 0) * weight
                if max_columns > 0:
                    column_losses, mask = _column_loss_values(row_result)
                    limit = min(len(column_losses), max_columns, len(mask))
                    for col_idx in range(limit):
                        if not mask[col_idx]:
                            continue
                        column_loss_sums[col_idx] += column_losses[col_idx] * weight
                        column_token_counts[col_idx] += weight

        summary = EvalSummary(
            loss_sums=overall_loss_sums,
            token_counts=overall_token_counts,
            column_loss_sums=column_loss_sums,
            column_token_counts=column_token_counts,
            source_label=source_label,
        )

    if was_training:
        model.train()
    return summary

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
        for legacy_key in ("use_carpet", "use_carpet2", "use_carpet3"):
            saved.pop(legacy_key, None)
        if "n_pos" not in saved:
            if "block_size" in saved:
                saved["n_pos"] = saved.pop("block_size")
            else:
                saved["n_pos"] = MODEL_GEOMETRY_DEFAULTS.n_pos
        legacy_xctx = bool(saved.pop("grce_xctx", False))
        if "n_xctx" not in saved:
            if legacy_xctx:
                saved["n_xctx"] = int(saved.get("n_grce", 0))
                saved["n_grce"] = 0
            else:
                saved["n_xctx"] = 0
        if "n_rope" not in saved:
            saved["n_rope"] = MODEL_GEOMETRY_DEFAULTS.n_rope
        if "n_query" not in saved:
            saved["n_query"] = MODEL_GEOMETRY_DEFAULTS.n_query
        if "use_gmlp" not in saved:
            saved["use_gmlp"] = MODEL_GEOMETRY_DEFAULTS.use_gmlp
        config = ModelGeometry(**saved)
        args.checkpoint_payload_override = payload
        args.tokenizer_json_override = payload.get("tokenizer_json")
        args.n_pos = config.n_pos
        if not getattr(args, "_block_size_defined", False):
            args.block_size = config.n_pos
        elif args.block_size > config.n_pos:
            if not getattr(args, "allow_oversize", False):
                raise ValueError(
                    "--block-size cannot exceed checkpoint --n-pos (use --allow-oversize)"
                )
            else:
                print(
                    color_text(
                        f"Warning: block_size {args.block_size} exceeds checkpoint n_pos {config.n_pos};"
                        " attention masks will be truncated to n_pos",
                        Colors.YELLOW,
                    )
                )
        args.n_layer = config.n_layer
        args.n_head = config.n_head
        args.n_width = config.n_width
        args.n_rope = config.n_rope
        args.n_grce = config.n_grce
        args.n_xctx = config.n_xctx
        args.n_query = config.n_query
        args.use_gmlp = getattr(config, "use_gmlp", MODEL_GEOMETRY_DEFAULTS.use_gmlp)
        args.vocab_size = config.vocab_size
        args.model_path_override = checkpoint_path
        args.log_path_override = checkpoint_path.with_suffix(".log")
        return

    inferred = ModelGeometry(
        vocab_size=args.vocab_size,
        n_pos=args.n_pos,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_width=args.n_width,
        n_rope=args.n_rope,
        n_grce=args.n_grce,
        n_xctx=args.n_xctx,
        use_gmlp=getattr(args, "use_gmlp", MODEL_GEOMETRY_DEFAULTS.use_gmlp),
        use_sane=getattr(args, "use_sane", MODEL_GEOMETRY_DEFAULTS.use_sane),
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
        self.article_separator_token_id: int | None = None
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

        article_token_id = None
        article_tokens = tokenizer.encode_ids("<|----|>")
        if len(article_tokens) == 1:
            article_token_id = article_tokens[0]

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
        self.article_separator_token_id = article_token_id
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
        used_tokens = max(0, used_tokens)
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
            article_separator_token_id=self.article_separator_token_id,
            align_articles=bool(getattr(self.args, "align_articles", False)),
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
            used_tokens = int(entry.get("used_train_tokens", 0) or 0)
            max_tokens = max(1, int(entry.get("max_train_tokens", 0) or 0))
            ratio = used_tokens // max_tokens
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
        baseline_steps = int(payload.get("train_step_index", 0) or 0)
        payload["loss_history"] = []
        payload["completed_cycles"] = 0
        payload["train_step_index"] = 0
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

    def cli_import(
        self,
        *,
        model: GRCEGPT,
        model_path: pathlib.Path,
        tokenizer_json: str,
        device: torch.device,
    ) -> int:
        source_path: pathlib.Path | None = getattr(self.args, "import_source", None)
        if source_path is None:
            raise ValueError("import command requires a source checkpoint path")
        if not source_path.exists():
            raise FileNotFoundError(f"Import checkpoint {source_path} not found")
        payload = torch.load(source_path, map_location=device, weights_only=False)
        metadata: dict[str, Any]
        if isinstance(payload, dict) and "model" in payload:
            metadata = dict(payload)
            source_state = upgrade_state_dict(payload["model"])
            src_cfg = payload.get("config")
            source_use_sane = bool(src_cfg.get("use_sane", False)) if isinstance(src_cfg, dict) else False
        else:
            metadata = {}
            state_dict = payload if isinstance(payload, dict) else payload
            source_state = upgrade_state_dict(state_dict)
            source_use_sane = False
        target_use_sane = bool(getattr(self.args, "use_sane", False))
        if source_use_sane and not target_use_sane:
            source_state = _strip_sane_parameters(source_state)
        load_summary = _load_checkpoint_state(
            model,
            source_state,
            allow_partial=True,
        )
        if not load_summary.get("success", False):
            _log_partial_checkpoint_warning(load_summary)
        reused = int(load_summary.get("reused", 0))
        total = int(load_summary.get("total", 0))
        metadata.pop("optimizer", None)
        metadata["model"] = model.state_dict()
        metadata["config"] = asdict(args_to_model_geometry(self.args))
        metadata["tokenizer_json"] = tokenizer_json or metadata.get("tokenizer_json")
        atomic_torch_save(metadata, model_path)
        print(
            color_text(
                (
                    f"Imported checkpoint from {source_path} -> {model_path}; "
                    f"reused {reused}/{total} tensors"
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
        log_offset = None
        ansi_offset = None
        global global_runtime_args
        global_runtime_args = self.args
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
            self._model_for_summary = None
            self._embed_params_for_summary = None
            self._tokens_for_summary = 0
            summary_line = f"Model: {model_path}"
            print(color_text(summary_line, Colors.CYAN))
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

            if self.args.command not in {"create", "import"} and self.args.tokenizer:
                print(
                    color_text(
                        "--tokenizer is only supported with the 'create' or 'import' commands; remove it and rerun.",
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

            tokenizer_needed = self.args.command not in {"create", "import", "corpus", "size"}
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
            if self.args.command in {"create", "import"}:
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
                _expected_sections(config, config.n_pos)
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
            chin_goal = non_emb_params * 20
            print(
                color_text(
                    f"Trainable model params: {total_params:,}; "
                    f"excl. embeddings: {non_emb_params:,}",
                    Colors.BLUE,
                )
            )
            tok_vecs = config.vocab_size
            emb_params = embedding_params
            print(
                color_text(
                    f"Learned token embeddings: {tok_vecs}; params={emb_params:,}",
                    Colors.BLUE,
                )
            )

            cmdline = " ".join(shlex.quote(arg) for arg in sys.argv)
            timestamp = datetime.now(timezone.utc).isoformat()
            log_file = log_path.open("a+", encoding="utf-8")
            log_file.seek(0, os.SEEK_END)
            log_offset = log_file.tell()
            log_file.write(f"\n[{timestamp}] {cmdline}\n")
            log_file.flush()
            if not self.args.no_ansi:
                ansi_path = log_path.with_suffix(".ansi")
                ansi_file = ansi_path.open("a+", encoding="utf-8")
                ansi_file.seek(0, os.SEEK_END)
                ansi_offset = ansi_file.tell()
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

            train_step_index = 0
            loss_history: List[Dict[str, float]] = []
            total_train_wall = 0.0
            total_train_tokens = 0
            payload = getattr(self.args, "checkpoint_payload_override", None)
            optimizer_state = None
            if payload is None and model_path.exists():
                payload = torch.load(
                    model_path,
                    map_location=device,
                    weights_only=False,  # checkpoints also store dataset offsets/counters
                )
            control_emb_expanded = False
            if payload is not None:
                saved_config = payload.get("config") if isinstance(payload, dict) else None
                source_use_sane = bool(saved_config.get("use_sane", False)) if isinstance(saved_config, dict) else False
                target_use_sane = bool(getattr(self.args, "use_sane", False))
                allow_partial_load = self.args.allow_shape_mismatch_load or (
                    target_use_sane and not source_use_sane
                )
                load_summary: dict[str, object] | None = None
                if isinstance(payload, dict) and "model" in payload:
                    self.args.completed_cycles = int(payload.get("completed_cycles", 0) or 0)
                    before_rows = _control_embedding_rows(payload.get("model", {}))
                    upgraded = upgrade_state_dict(payload["model"])
                    control_emb_expanded = bool(
                        isinstance(before_rows, int)
                        and before_rows < CONTROL_EMBEDDING_ROWS
                    )
                    if source_use_sane and not target_use_sane:
                        upgraded = _strip_sane_parameters(upgraded)
                    payload["model"] = upgraded
                    load_summary = _load_checkpoint_state(
                        model,
                        upgraded,
                        allow_partial=allow_partial_load,
                    )
                    if load_summary["success"]:
                        if self.args.checkpoint_optimizer:
                            optimizer_state = payload.get("optimizer")
                            if control_emb_expanded:
                                optimizer_state = None
                    else:
                        optimizer_state = None
                        _log_partial_checkpoint_warning(load_summary)
                    train_step_index = int(payload.get("train_step_index", 0) or 0)
                    loss_history = list(payload.get("loss_history", []))
                    total_train_wall = float(payload.get("train_wall_seconds", 0.0))
                    total_train_tokens = int(payload.get("total_train_tokens", 0) or 0)
                    prompt_registry = PromptRegistry(
                        tokenizer,
                        payload.get("prompts"),
                    )
                    if "tokenizer_json" in payload:
                        self.args.tokenizer_json_override = payload.get("tokenizer_json")
                else:
                    before_rows = _control_embedding_rows(payload) if hasattr(payload, "get") else None
                    upgraded_payload = upgrade_state_dict(
                        _strip_sane_parameters(payload)
                        if (source_use_sane and not target_use_sane and hasattr(payload, "items"))
                        else payload
                    )
                    if isinstance(before_rows, int) and before_rows < CONTROL_EMBEDDING_ROWS:
                        control_emb_expanded = True
                    load_summary = _load_checkpoint_state(
                        model,
                        upgraded_payload,
                        allow_partial=allow_partial_load,
                    )
                    if not load_summary["success"]:
                        optimizer_state = None
                        _log_partial_checkpoint_warning(load_summary)
                print(color_text(f"Loaded existing model from {model_path}", Colors.YELLOW))
                hours = total_train_wall / 3600.0
                days = hours / 24.0
                print(
                    color_text(
                        f"Total training so far: {train_step_index} steps, {hours:.2f} hours ({days:.2f} days)",
                        Colors.YELLOW,
                    )
                )
            if self.args.command == "import":
                if tokenizer_json is None:
                    raise RuntimeError("Tokenizer JSON is required for the import command")
                return self.cli_import(
                    model=model,
                    model_path=model_path,
                    tokenizer_json=tokenizer_json,
                    device=device,
                )
            if getattr(self.args, "_cycles_is_delta", False):
                delta = int(getattr(self.args, "_cycles_delta", 0))
                base_cycles = int(getattr(self.args, "completed_cycles", 0))
                self.args.cycles = base_cycles + delta

            if self.args.command == "create":
                checkpoint_payload = {
                    "model": model.state_dict(),
                    "train_step_index": 0,
                    "loss_history": [],
                    "config": asdict(args_to_model_geometry(self.args)),
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
                    use_decode_mode=self.args.generate_with_decode,
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
                    block_size=self.args.block_size,
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
                    summary = run_eval_layout(
                        args=self.args,
                        dataset=dataset,
                        tokenizer=tokenizer,
                        model=model,
                        block_size=self.args.block_size,
                        start_pos=self.args.eval_start,
                        custom_text=custom_text,
                        verbose=self.args.eval_verbose,
                        align_rng=None,
                    )
                    _print_eval_summary("[eval custom]", self.args.layout, [summary])
                    return
                rand_runs = max(0, int(getattr(self.args, "eval_random", 0)))
                if rand_runs > 0:
                    total = len(dataset.test_tokens)
                    if total <= 0:
                        raise ValueError("Test corpus is empty; cannot run random evaluations")
                    base_seed = max(0, int(getattr(self.args, "rng_seed", 0)))
                    rng = random if base_seed > 0 else random.Random()
                    window = max(1, total - (self.args.block_size + 1))
                    batch_cap = max(1, int(self.args.batch_size))
                    layout_probe = BatchLayout(
                        self.args.layout,
                        batch_size=self.args.batch_size,
                        block_size=self.args.block_size,
                    )
                    _log_layout_warnings(self.args, layout_probe)
                    max_positions = max(
                        (row.total_positions() for row in layout_probe.rows), default=0
                    )
                    if max_positions <= 0:
                        raise ValueError("Layout does not contain any token positions to evaluate")
                    offsets = [rng.randint(0, window - 1) for _ in range(rand_runs)]
                    summaries: list[EvalSummary] = []
                    total_batches = (rand_runs + batch_cap - 1) // batch_cap
                    for batch_idx in range(total_batches):
                        start_idx = batch_idx * batch_cap
                        batch_offsets = offsets[start_idx : start_idx + batch_cap]
                        if self.args.eval_verbose or True:
                            print(
                                color_text(
                                    f"[eval random batch {batch_idx + 1}/{total_batches} - {100.0*(batch_idx + 1) / total_batches:.2f}%]",
                                    Colors.BLUE,
                                )
                            )
                        batch_inputs, batch_targets, batch_label, batch_pretty = _prepare_eval_batch_tokens(
                            self.args,
                            dataset,
                            tokenizer,
                            token_length=max_positions,
                            start_positions=batch_offsets,
                            align_rng=rng if self.args.align_articles else None,
                        )
                        summary = run_eval_layout(
                            args=self.args,
                            dataset=dataset,
                            tokenizer=tokenizer,
                            model=model,
                            block_size=self.args.block_size,
                            start_pos=0,
                            custom_text=None,
                            verbose=self.args.eval_verbose,
                            align_rng=None,
                            batch_inputs=batch_inputs,
                            batch_targets=batch_targets,
                            batch_label=batch_label,
                            batch_pretty=batch_pretty,
                            use_sampled_weight=True,
                        )
                        summaries.append(summary)
                    _print_eval_summary("[eval random]", self.args.layout, summaries)
                    return
                summary = run_eval_layout(
                    args=self.args,
                    dataset=dataset,
                    tokenizer=tokenizer,
                    model=model,
                    block_size=self.args.block_size,
                    start_pos=self.args.eval_start,
                    custom_text=None,
                    verbose=self.args.eval_verbose,
                    align_rng=None,
                )
                _print_eval_summary("[eval]", self.args.layout, [summary])
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
                    block_size=self.args.block_size,
                    batch_size=self.args.batch_size,
                    device=device,
                    n_pos=self.args.n_pos,
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
            if self.args.command == "try":
                cycle_end = cycle_start
            else:
                cycle_end = max(completed_cycles, self.args.cycles)
            rng_cycle_only = bool(getattr(self.args, "rng_cycle_only", False))
            for cycle in range(cycle_start, cycle_end + 1):
                base_seed = max(0, int(getattr(self.args, "rng_seed", 0)))
                if base_seed > 0:
                    seed_total_tokens = total_train_tokens
                    cycle_seed = _derive_cycle_token_seed(
                        base_seed,
                        cycle,
                        seed_total_tokens,
                        cycle_only=rng_cycle_only,
                    )
                    _apply_global_rng_seed(cycle_seed)
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
                token_fragment = f"{total_train_tokens:,} tokens"
                param_fragment = f"{non_emb_params:,} params"
                if chin_goal > 0:
                    pct = (total_train_tokens / chin_goal) * 100.0
                    chin_fragment = f"{pct:.2f}% of 20x"
                else:
                    chin_fragment = "∞% of 20x"
                model_line = (
                    f"Model: {model_path} ({token_fragment} / {param_fragment} = {chin_fragment})"
                )
                print(color_text(model_line, Colors.CYAN))
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
                cycles_pct = (per_run_idx / max(1, self.args.cycles)) * 100.0 if self.args.cycles > 0 else 0.0
                print(
                    color_text(
                        f"[{label}] Training Cycle {per_run_idx}/{self.args.cycles} ({cycles_pct:.2f}%). "
                        f"Total training so far: {train_step_index} steps, {hours:.2f} hours ({days:.2f} days)",
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
                    train_step_index,
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
                    self.args.block_size,
                    self.args.batch_size,
                    self.args.eval_interval,
                    train_step_index,
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
                        "train_step_index": train_step_index,
                        "loss_history": loss_history,
                        "config": asdict(args_to_model_geometry(self.args)),
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
            keep_logs = bool(getattr(self.args, "log_all", False)) or bool(getattr(self.args, "checkpoint_dirty", False))
            if log_file is not None:
                if not keep_logs and log_offset is not None:
                    try:
                        log_file.flush()
                        log_file.seek(log_offset)
                        log_file.truncate()
                    except OSError:
                        pass
                log_file.close()
            if ansi_file is not None:
                if not keep_logs and ansi_offset is not None:
                    try:
                        ansi_file.flush()
                        ansi_file.seek(ansi_offset)
                        ansi_file.truncate()
                    except OSError:
                        pass
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

    base_seed = max(0, int(getattr(args, "rng_seed", DEFAULTS.rng_seed)))
    if base_seed > 0:
        _apply_global_rng_seed(base_seed)
    else:
        torch.manual_seed(42)
        random.seed(time.time())

    # otherwise: run the big "default" main
    rt = Runtime(args)
    rt.start_timeout(args.timeout)
    return rt.big_fat_old_main()

if __name__ == "__main__":
    sys.exit(grce_main(cli_args))
