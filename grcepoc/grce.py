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

* :class:`ModelConfig` – dataclass that defines the model geometry and feeds
into tokenizer/model builders, :func:`describe_model_size`, and
:func:`grce_main`.
* :func:`grce_cli_args` – constructs the CLI parser; invoked at startup and by
external tooling to mirror the binary interface. Its result is consumed by
:func:`grce_main`.
* :func:`train_model` – the main training loop used by :func:`grce_main`. It
handles batching, diagnostics, and logging.
* :func:`evaluate_single_batch` – computes evaluation metrics from a single
forward pass that mirrors the training row-type composition.
* :func:`describe_model_size` – backs the ``size`` subcommand by combining
  :class:`ModelConfig` metadata with :func:`compute_row_type_counts`.

Call tree (simplified)::

    grce_cli_args
        └── grce_main
            ├── train_model
            │     └── evaluate_single_batch
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

SPECIAL_TOKENS = """
<|----|> <|//|> <|tokipona:> <|english:> </english:> </tokipona:>
<|lq:> <:lq|> <|hq:> <:hq|> <|!:> <:!|> <|?:> <:?|> <|*:> <:*|>
<|-:> <:-|> <|=:> <:=|> <|/:> <:/|> <|@:> <:@|> <|think|>
<|p7|> <|p6|> <|p5|> <|p4|> <|p3|> <|p2|> <|p1|> <|p0|>
""".split()


# -----------------------------------------------------------------------------
# GRCE Model Configuration
# -----------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Holds the GPT+GRCE+XCTX model geometry.

    Instances are created in :func:`grce_cli_args` and threaded through the
    tokenizer builder, :func:`describe_model_size`, and :func:`grce_main`.
    """

    vocab_size: int = 3000  # GPT-2 base supports ~50k merges; we stay small for the PoC.
    block_size: int = 256   # GPT-2 base uses 1024 tokens.
    n_layer: int = 8        # GPT-2 base uses 12 layers.
    n_head: int = 6         # GPT-2 base uses 12 attention heads.
    n_embd: int = 384       # GPT-2 base uses 768 embedding dims.
    n_grce: int = 64        # Narrow GRCE context dims.
    n_xctx: int = 720       # Wide XCTX context dims.

    # FIXME: these should only be part of Settings, not ModelConfig -> remove later
    dropout: float = 0.05
    detach_span: int = 0    # Detach gradients every N positions (0 disables detaching).
    detach_context: bool = True  # Whether to detach recurring context when span triggers.
    detach_layer: int = -1       # Layer index (1-based) after which to detach Transformer grads.

MODEL_CONFIG_DEFAULTS = ModelConfig()


@dataclass
class Settings:
    """Holds the GRCE/XCTX model geometry and all other runtime settings.

    (Most users of ModelConfig should actually be moved to using Settings instead)
    """

    # The grce_cli_args() return value. Initialize this field
    # via the Settings constructor and Settings.__post_init__() will
    # will populate the other fields here from that argparse namespace.
    cli_args: argparse.Namespace | None = None

    # Model Geometry
    vocab_size: int = MODEL_CONFIG_DEFAULTS.vocab_size
    block_size: int = MODEL_CONFIG_DEFAULTS.block_size
    n_layer: int = MODEL_CONFIG_DEFAULTS.n_layer
    n_head: int = MODEL_CONFIG_DEFAULTS.n_head
    n_embd: int = MODEL_CONFIG_DEFAULTS.n_embd
    n_grce: int = MODEL_CONFIG_DEFAULTS.n_grce
    n_xctx: int = MODEL_CONFIG_DEFAULTS.n_xctx

    # Additional non-geometry "pseudo" model args
    corpus: str = "simplerwiki"
    extra_tags: tuple[str] = ()

    # Training Loop
    steps: int = 100
    cycles: int = 100
    batch_size: int = 32
    eval_interval: int = 10
    _block_length_arg: int | None = None

    # Training Details
    dropout: float = 0.05
    detach_span: int = 0
    detach_context: bool = False
    detach_layer: int = -1

    # Logging and diagnostics
    escape_newline_tokens: bool = True
    show_train_loss_details: bool = False
    show_test_loss_details: bool = True

    @property
    def block_length(self):
        if self._block_length_arg is not None:
            return self._block_length_arg
        return self.block_size

    def __post_init__(self):
        if self.cli_args is None: return
        args = self.cli_args

        self.vocab_size = args.vocab_size
        self.block_size = args.block_size
        self.n_layer = args.n_layer
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.n_grce = args.n_grce
        self.n_xctx = args.n_xctx

        self.corpus = args.corpus
        self.extra_tags = tuple(args.tag)

        self.steps = args.steps
        self.cycles = args.cycles
        self.batch_size = args.batch_size
        self.eval_interval = args.eval_interval
        self._block_length_arg = args.block_length

        self.dropout = args.dropout
        self.detach_span = args.detach_span
        self.detach_context = not args.no_detach_ctx
        self.detach_layer = args.detach_layer

        self.escape_newline_tokens = not args.no_escape_newline_tokens
        self.show_train_loss_details = args.train_loss_details
        self.show_test_loss_details = not args.no_test_loss_details

SETTINGS_DEFAULTS = Settings()


# -----------------------------------------------------------------------------
# GRCE Model Helper Functions
# -----------------------------------------------------------------------------

def compute_row_type_counts(batch_size: int) -> dict[str, int]:
    """Return the deterministic row-type counts for a batch size."""

    total = max(0, int(batch_size))
    if total == 0:
        return {
            "n_decode": 0,
            "n_encode": 0,
            "n_recode": 0,
            "n_noxctx": 0,
            "n_puxctx": 0,
            "n_noattn": 0,
            "n_puattn": 0,
            "n_normal": 0,
            "n_total": 0,
        }

    counts: dict[str, int] = {
        "n_decode": 0,
        "n_encode": 0,
        "n_recode": 0,
        "n_noxctx": 0,
        "n_puxctx": 0,
        "n_noattn": 0,
        "n_puattn": 0,
        "n_normal": 0,
    }
    remaining = total
    special_order = [
        "n_encode",
        "n_recode",
        "n_noxctx",
        "n_puxctx",
        "n_noattn",
        "n_puattn",
    ]
    for key in special_order:
        if remaining <= 0:
            break
        counts[key] = 1
        remaining -= 1
    if remaining > 0:
        decode = max(1, remaining // 2)
        decode = min(decode, remaining)
        counts["n_decode"] = decode
        remaining -= decode
    counts["n_normal"] = remaining
    counts["n_total"] = total
    row_sum = sum(
        value for key, value in counts.items() if key.startswith("n_") and key != "n_total"
    )
    if row_sum != total:
        raise ValueError(
            f"Row-type composition mismatch: sum={row_sum} differs from n_total={total}"
        )
    return counts


def build_row_type_template(
    row_counts: dict[str, int],
    batch_size: int,
    *,
    context_enabled: bool,
    xctx_enabled: bool,
) -> list[str]:
    """Expand row counts into a shuffled template for a batch.

    :func:`train_model` calls this helper before mask creation so every batch
    follows the README-specified composition.
    """

    template: list[str] = []

    def allocate(name: str, supported: bool) -> None:
        count = int(max(0, row_counts.get(f"n_{name}", 0)))
        if count <= 0:
            return
        if not supported:
            return
        template.extend([name] * count)

    allocate("decode", context_enabled)
    allocate("noxctx", context_enabled and xctx_enabled)
    allocate("puxctx", context_enabled and xctx_enabled)
    allocate("noattn", True)
    allocate("puattn", True)
    allocate("encode", True)
    allocate("recode", True)
    normal_count = max(0, int(row_counts.get("n_normal", 0)))
    template.extend(["normal"] * normal_count)
    if len(template) > batch_size:
        raise ValueError(
            f"Row-type template exceeded batch size: built {len(template)} entries for n_total={batch_size}"
        )
    if len(template) < batch_size:
        template.extend(["normal"] * (batch_size - len(template)))
    return template


def build_row_type_masks(
    row_types: Sequence[str],
    block_length: int,
    device: torch.device,
    *,
    context_enabled: bool,
    xctx_enabled: bool,
) -> SpecialRowMasks:
    """Create the per-row masks for a row template.

    The returned :class:`SpecialRowMasks` structure is consumed by
    :func:`train_model` to disable GRCE, XCTX, or attention for specific rows.
    """
    batch_size = len(row_types)
    context_special_rows: set[int] = set()
    context_disabled_mask = (
        torch.zeros(batch_size, dtype=torch.bool, device=device) if context_enabled else None
    )
    context_bias_disabled_mask: torch.Tensor | None = None
    xctx_disabled_mask = (
        torch.zeros(batch_size, dtype=torch.bool, device=device) if xctx_enabled else None
    )
    xctx_bias_disabled_mask: torch.Tensor | None = None
    context_dropout_positions = None
    attention_disabled_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    attention_dropout_positions = None
    encode_rows = torch.zeros(batch_size, dtype=torch.bool, device=device)
    recode_rows = torch.zeros(batch_size, dtype=torch.bool, device=device)
    recode_boundaries = torch.full(
        (batch_size,),
        -1,
        dtype=torch.long,
        device=device,
    )

    def ensure_mask(mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        return mask
    def ensure_tensor(tensor: torch.Tensor | None, *, dtype, fill) -> torch.Tensor:
        if tensor is None:
            return torch.full((batch_size,), fill, dtype=dtype, device=device)
        return tensor

    for idx, row_type in enumerate(row_types):
        if row_type in {"decode", "encode"} and context_enabled and context_disabled_mask is not None:
            context_disabled_mask[idx] = True
            context_special_rows.add(idx)
            context_bias_disabled_mask = ensure_mask(context_bias_disabled_mask)
            context_bias_disabled_mask[idx] = True
        if row_type in {"decode", "encode"} and xctx_enabled and xctx_disabled_mask is not None:
            xctx_disabled_mask[idx] = True
            context_special_rows.add(idx)
            xctx_bias_disabled_mask = ensure_mask(xctx_bias_disabled_mask)
            xctx_bias_disabled_mask[idx] = True
        if row_type == "encode":
            encode_rows[idx] = True
        if row_type == "recode":
            recode_rows[idx] = True
            recode_boundaries[idx] = max(1, block_length // 2)
        elif row_type == "puxctx" and xctx_enabled:
            context_dropout_positions = ensure_tensor(
                context_dropout_positions,
                dtype=torch.long,
                fill=-1,
            )
            context_dropout_positions[idx] = random.randrange(max(1, block_length))
            context_special_rows.add(idx)
        elif row_type == "noattn":
            attention_disabled_mask[idx] = True
            context_special_rows.add(idx)
        elif row_type == "puattn":
            attention_dropout_positions = ensure_tensor(
                attention_dropout_positions,
                dtype=torch.long,
                fill=-1,
            )
            attention_dropout_positions[idx] = random.randrange(max(1, block_length))
            context_special_rows.add(idx)
    if context_disabled_mask is not None and not context_disabled_mask.any():
        context_disabled_mask = None
    if context_bias_disabled_mask is not None and not context_bias_disabled_mask.any():
        context_bias_disabled_mask = None
    if xctx_disabled_mask is not None and not xctx_disabled_mask.any():
        xctx_disabled_mask = None
    if xctx_bias_disabled_mask is not None and not xctx_bias_disabled_mask.any():
        xctx_bias_disabled_mask = None
    if context_dropout_positions is not None and (context_dropout_positions < 0).all():
        context_dropout_positions = None
    if not attention_disabled_mask.any():
        attention_disabled_mask = None
    if attention_dropout_positions is not None and (attention_dropout_positions < 0).all():
        attention_dropout_positions = None
    if not encode_rows.any():
        encode_rows = None
    if not recode_rows.any():
        recode_rows = None
        recode_boundaries = None

    return SpecialRowMasks(
        context_special_rows,
        context_disabled_mask,
        context_bias_disabled_mask,
        xctx_disabled_mask,
        xctx_bias_disabled_mask,
        context_dropout_positions,
        attention_disabled_mask,
        attention_dropout_positions,
        encode_rows,
        recode_rows,
        recode_boundaries,
    )


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
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Sequence
from collections import defaultdict


def parse_range_arg(value: str) -> tuple[int, int]:
    parts = value.replace(" ", "").split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid range '{value}'. Expected format START-END.")
    start, end = int(parts[0]), int(parts[1])
    if end < start:
        raise ValueError(f"Range end {end} is smaller than start {start}.")
    return start, end


def grce_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and return the populated namespace.

    Used by the ``if __name__ == '__main__'`` entry point and by tooling that
    wants to mirror the CLI behavior without invoking the binary. The result
    is passed directly to :func:`grce_main`.
    """
    defaults = SETTINGS_DEFAULTS
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
        "--corpus",
        type=str,
        default=defaults.corpus,
        help="Dataset base name; expects data/<name>-train.txt.gz and ...-test.txt.gz.",
    )
    generic.add_argument(
        "--data",
        type=str,
        default="data",
        help="Directory containing <corpus>-train.txt.gz and <corpus>-test.txt.gz",
    )
    generic.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
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
        default=defaults.vocab_size,
        help="Total vocabulary size for the tokenizer (including special tokens)",
    )
    model_group.add_argument(
        "--block-size",
        type=int,
        default=defaults.block_size,
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
        default=defaults.n_layer,
        help="Number of transformer blocks (GPT-2 base uses 12).",
    )
    model_group.add_argument(
        "--n-head",
        type=int,
        default=defaults.n_head,
        help="Number of attention heads per block (GPT-2 base uses 12).",
    )
    model_group.add_argument(
        "--n-embd",
        type=int,
        default=defaults.n_embd,
        help="Embedding/hidden dimension (GPT-2 base uses 768); must be a multiple of n_head.",
    )
    model_group.add_argument(
        "--n-grce",
        type=int,
        default=defaults.n_grce,
        help="Dimension of the recurrent GRCE context; use 0 to disable the channel.",
    )
    model_group.add_argument(
        "--n-xctx",
        type=int,
        default=defaults.n_xctx,
        help=(
            "Dimension of the wide (layer-partitioned) context channel; must be a multiple of n_layer"
        ),
    )
    model_group.add_argument(
        "--tiny",
        action="store_true",
        help=(
            "Shortcut for --vocab-size 500 --batch-size 12 --block-size 6 --n-layer 3 --n-head 2 "
            "--n-embd 8 --n-grce 4 --n-xctx 9 --steps 2 --cycles 1 --eval-interval 1 --corpus simplestwiki"
        ),
    )

    training_group = parser.add_argument_group("Training schedule")
    training_group.add_argument("--steps", type=int, default=defaults.steps, help="Training steps per cycle")
    training_group.add_argument(
        "--cycles",
        type=int,
        default=defaults.cycles,
        help="Repeat the full training/eval/update cycle N times.",
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        default=defaults.batch_size,
        help="Number of sequences per optimization step.",
    )
    training_group.add_argument(
        "--detach-span",
        type=int,
        default=defaults.detach_span,
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
        default=defaults.dropout,
        help="Dropout probability inside attention/FFN blocks.",
    )
    training_group.add_argument(
        "--detach-layer",
        type=int,
        default=-1,
        help="If >0, detach gradients after this Transformer layer (1-based index).",
    )
    training_group.add_argument(
        "--reset-prompt-each-cycle",
        action="store_true",
        help=(
            "Rebuild the prompt queue at the start of every training cycle; default keeps cycling "
            "through prompts across cycles"
        ),
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
        "--train-loss-details",
        action="store_true",
        help="Show the per-row loss columns in the live log",
    )
    logging_group.add_argument(
        "--no-test-loss-details",
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
        "--import-model",
        type=pathlib.Path,
        help="Initialize from another checkpoint when creating a new model",
    )
    import_group.add_argument(
        "--trim-model",
        action="store_true",
        help="Allow importing into a smaller model by dropping overflow",
    )
    import_group.add_argument(
        "--drop-layers",
        type=str,
        default="",
        help="Comma-separated layer numbers (1-indexed) to remove during import",
    )
    import_group.add_argument(
        "--add-layers",
        type=str,
        default="",
        help="Comma-separated layer numbers (1-indexed) to insert during import",
    )
    import_group.add_argument(
        "--pt",
        type=pathlib.Path,
        help="Load an explicit checkpoint file for inference/debugging commands",
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

    size_parser = subparsers.add_parser(
        "size",
        help="Print parameter breakdown for the configured model and exit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    size_parser.set_defaults(command="size")
    size_parser.add_argument(
        "--check",
        action="store_true",
        help="Instantiate the model and verify the analytic counts",
    )
    size_parser.add_argument(
        "--estimate",
        action="store_true",
        help="Append a dominant-term estimate section",
    )

    corpus_parser = subparsers.add_parser(
        "corpus",
        help="Manage tokenizer and cached corpora",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    corpus_parser.add_argument(
        "--init-tokenizer",
        action="store_true",
        dest="corpus_init_tokenizer",
        help="Train or refresh the tokenizer JSON for this corpus",
    )
    corpus_parser.add_argument(
        "--init",
        action="store_true",
        dest="corpus_init",
        help="Regenerate the cached token files (requires an existing tokenizer JSON)",
    )
    corpus_parser.add_argument(
        "--print-train",
        dest="corpus_print_train",
        metavar="START-END",
        help="Print a START-END token range from the train split",
    )
    corpus_parser.add_argument(
        "--print-test",
        dest="corpus_print_test",
        metavar="START-END",
        help="Print a START-END token range from the test split",
    )
    corpus_parser.set_defaults(command="corpus")

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

    create_parser = subparsers.add_parser(
        "create",
        help="Create a new checkpoint with random weights and exit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    create_parser.set_defaults(command="create")

    block_length_flag = flag_present("--block-length")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit(
            1,
            "\nPlease specify a command (train, report, test, size, corpus, create, or prompts).\n",
        )
    if args.command == "corpus":
        has_corpus_action = bool(
            getattr(args, "corpus_init", False)
            or getattr(args, "corpus_print_train", None)
            or getattr(args, "corpus_print_test", None)
        )
        if not has_corpus_action:
            parser.error("corpus command requires --init and/or --print-* options")
    if args.command == "create" and args.pt:
        parser.error("--pt cannot be combined with the create command")
    if args.tiny:
        if not flag_present("--vocab-size"):
            args.vocab_size = 500
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
        if not flag_present("--corpus"):
            args.corpus = "simplestwiki"
    if args.block_length is None:
        args.block_length = args.block_size
    if args.block_length <= 0:
        parser.error("--block-length must be positive")
    if args.block_length > args.block_size:
        parser.error("--block-length must be <= --block-size")
    args._block_length_defined = block_length_flag
    return args


if __name__ == "__main__":
    cli_args = grce_cli_args(sys.argv)


# -----------------------------------------------------------------------------
# Lightweight (non-Torch) Library Components
# -----------------------------------------------------------------------------

def load_text_file(path: pathlib.Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Provide a text file path.")
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8")


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class Colors:
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
    prefix = ""
    if bold:
        prefix += Colors.BOLD
    if underline:
        prefix += Colors.UNDERLINE
    return f"{prefix}{color}{text}{Colors.RESET}"


def normalize_prompt(text: str) -> str:
    """Map placeholder characters back to literal spaces/newlines."""

    return text.replace(FANCY_SPACE, " ").replace(FANCY_ENTER, "\n")



def prompt_needs_boundary(text: str) -> bool:
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
    def __init__(self):
        self.gpu_sampler = GpuUtilSampler(interval_s=0.2)
        self.wall_secs = 0.0
        self.cpu_secs = 0.0
        self.gpu_secs = 0.0
        self.gpu_mem = 0.0
        self.wall_start = None
        self.cpu_start = None
        self.gpu_start = None

    def start(self):
        assert self.wall_start is None
        self.wall_start = time.time()
        self.cpu_start = time.process_time()
        self.gpu_sampler.start()
        return self

    def stop(self):
        assert self.wall_start is not None
        self.wall_secs += time.time() - self.wall_start
        self.cpu_secs += time.process_time() - self.cpu_start
        self.gpu_sampler.stop()
        self.gpu_secs += self.gpu_sampler.busy_time_s
        self.gpu_mem = max(self.gpu_mem, self.gpu_sampler.max_mem_util)
        self.wall_start = None
        self.cpu_start = None
        return self

    def __str__(self):
        return f"{self.wall_secs:.2f}s / {self.cpu_secs:.2f}s / {self.gpu_secs:.2f}s / {self.gpu_mem:.2f}%"


# -----------------------------------------------------------------------------
# GRCE Model Size Information
# -----------------------------------------------------------------------------

def _get_inner_xctx_width(config: ModelConfig) -> int:
    return min(config.n_embd // 2, config.n_xctx // 2, max(config.n_embd // 4, config.n_xctx // config.n_layer))

def _build_geometry(config: ModelConfig, block_size: int) -> list[tuple[str, str, int]]:
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

def _expected_sections(config: ModelConfig, block_size: int) -> list[tuple[str, str, list[dict]]]:
    V = config.vocab_size
    B = block_size
    L = max(1, config.n_layer)
    H = max(1, config.n_head)
    E = config.n_embd
    G = config.n_grce
    X = config.n_xctx

    sections: list[tuple[str, str, list[dict]]] = []

    global_items = [
        {"label": "token embeddings", "count": V * E, "formula": "V * E"},
        {"label": "position embeddings", "count": B * E, "formula": "B * E"},
        {"label": "special embeddings", "count": 0 * E, "formula": "0 * E"},
        {"label": "grce embeddings", "count": 0 * G, "formula": "0 * G"},
    ]
    sections.append(("embeddings", "Embeddings", global_items))

    transformer_items = [
        {
            "label": "attn qkv",
            "count": 3 * L * (E * E + E),
            "formula": "3 * L * (E * E + E)",
        },
        {
            "label": "attn proj",
            "count": L * (E * E + E),
            "formula": "L * (E * E + E)",
        },
        {
            "label": "ffn fc1",
            "count": L * (4 * E * E + 4 * E),
            "formula": "L * (4*E*E + 4*E)",
        },
        {
            "label": "ffn fc2",
            "count": L * (4 * E * E + E),
            "formula": "L * (4*E*E + E)",
        },
    ]
    sections.append(("transformer", "Transformer", transformer_items))

    if G > 0:
        grce_items = [
            {
                "label": "samplers",
                "count": config.n_layer * (2 * E + E * G + G),
                "formula": "L * (2*E + E*G + G)",
            },
            {
                "label": "mlp",
                "count": 8 * G * G + 9 * G,
                "formula": "8*G*G + 9*G",
            },
            {
                "label": "bias",
                "count": config.n_layer * (G * E + E),
                "formula": "L * (G*E + E)",
            },
        ]
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
                "count": config.n_layer * (X * chunk + chunk + chunk * E + E),
                "formula": "L * (X*(X/L) + (X/L) + (X/L)*E + E)",
            },
        ]
    else:
        xctx_items = []
    sections.append(("xctx", "XCTX Channel", xctx_items))

    # Placeholder for summary, filled later
    return sections


def _append_summary_section(sections: list[tuple[str, str, list[dict]]]) -> list[tuple[str, str, list[dict]]]:
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
    print(color_text("Model Geometry", Colors.CYAN, bold=True))
    for var, desc, value in geometry:
        print(f"  {var} ({desc:<17s}): {value}")


def _print_section(title: str, items: list[dict]) -> int:
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


def _flatten_expected(sections: list[tuple[str, str, list[dict]]]) -> dict[tuple[str, str], int]:
    mapping: dict[tuple[str, str], int] = {}
    for key, _title, items in sections:
        if key == "summary":
            continue
        for entry in items:
            mapping[(key, entry["label"])] = entry["count"]
    return mapping


def _compute_actual_counts(config: ModelConfig) -> dict[tuple[str, str], int]:
    model = GRCEGPT(config)
    counts: dict[tuple[str, str], int] = {}

    counts[("global", "token embeddings")] = _module_param_count(model.core.tok_emb)
    counts[("global", "position embeddings")] = _module_param_count(model.core.pos_emb)

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
        key = "xctx" if channel.is_xctx else "grce"
        breakdown = channel.parameter_breakdown()
        for label, value in breakdown.items():
            counts[(key, label)] = counts.get((key, label), 0) + value
    return counts


def _dominant_estimates(config: ModelConfig) -> list[tuple[str, int, str]]:
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
    settings: ModelConfig,
    *,
    check: bool = False,
    estimate: bool = False,
) -> None:
    """Emit the ``size`` subcommand report.

    print the standard parameter breakdown based on :class:`ModelConfig`.
    Called exclusively from :func:`grce_main`.
    """
    print()
    geometry = _build_geometry(settings, settings.block_size)
    sections = _append_summary_section(_expected_sections(settings, settings.block_size))
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
        for label, count, formula in _dominant_estimates(settings):
            print(f"  {label:<20} {count:>15,}  {formula}")

    if check:
        expected_map = _flatten_expected(sections)
        actual_map = _compute_actual_counts(settings)
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

def grce_cli_size(args: argparse.Namespace):
    assert args.command == "size"
    return grce_cmd_size(
        Settings(args),
        check=getattr(args, "check", False),
        estimate=getattr(args, "estimate", False),
    )

if __name__ == "__main__":
    # run it here when not in --check mode, and run it later if we need torch for --check
    # on some builds it can take 5 seconds or longer to import torch, so we early-exit
    # on the "size" sub-command here so it prints the model size right away without that delay
    if cli_args.command == "size" and not getattr(cli_args, "check", False):
        sys.exit(grce_cli_size(cli_args))


# -----------------------------------------------------------------------------
# GRCE Library Components
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import string
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

FANCY_SPACE = "\u2423"  # Open Box symbol for visible spaces
FANCY_ENTER = "\u23CE "  # Return symbol for visible newlines
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


# Local GPT2 tokenizer adapter (no huggingface dependency)
class GPT2TokenizerFast:
    """Lightweight adapter around ``tokenizers.Tokenizer`` used by GRCE.

    Instances are created in :func:`grce_main` to build tokenizers without the
    full Hugging Face dependency and are consumed via
    :class:`GPT2TokenizerWrapper`.
    """
    def __init__(self, tokenizer_file=None, tokenizer_object=None):
        if tokenizer_file is not None:
            self._tokenizer = Tokenizer.from_file(tokenizer_file)
        elif tokenizer_object is not None:
            self._tokenizer = tokenizer_object
        else:
            raise ValueError("Either tokenizer_file or tokenizer_object must be provided")
        self._extra_special_tokens: list[str] = []

    @property
    def unk_token(self) -> str:
        return "<|unk|>"

    def _normalize_special_tokens(self, tokens):
        if isinstance(tokens, dict):
            tokens = tokens.get("additional_special_tokens", []) or []
        if isinstance(tokens, str):
            tokens = [tokens]
        return list(tokens or [])

    def add_special_tokens(self, tokens):
        entries = self._normalize_special_tokens(tokens)
        added = []
        for tok in entries:
            if tok not in self._extra_special_tokens:
                self._extra_special_tokens.append(tok)
                added.append(tok)
        if added:
            self._tokenizer.add_special_tokens(added)
        return len(entries)

    def get_vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def __len__(self) -> int:
        return self.get_vocab_size()

    @property
    def all_special_ids(self) -> list[int]:
        ids: list[int] = []
        for token in self._extra_special_tokens:
            tok_id = self._tokenizer.token_to_id(token)
            if tok_id is not None:
                ids.append(tok_id)
        return ids

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(
        self,
        ids: list[int] | torch.Tensor,
        *,
        clean_up_tokenization_spaces: bool = True,
        skip_special_tokens: bool = False,
    ) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, "tokenizer.json")
        self._tokenizer.save(file)

def default_prompt_entries() -> list[tuple[str, str]]:
    return [(prompt, expected) for prompt, expected in PROMPT_GOALS]


def serialized_prompts(entries: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"prompt": prompt, "expected": expected} for prompt, expected in entries]


def empty_prompt_state(entries: list[tuple[str, str]] | None = None) -> dict:
    prompts = entries if entries is not None else default_prompt_entries()
    serialized = serialized_prompts(prompts)
    return {
        "status": [0] * len(serialized),
        "completed": [False] * len(serialized),
        "queue": None,
        "prompts": serialized,
    }

def build_prompt_state(
    entries: list[tuple[str, str]], statuses: list[int] | None = None
) -> dict:
    serialized = serialized_prompts(entries)
    total = len(serialized)
    cleaned: list[int] = []
    if statuses is None:
        cleaned = [0] * total
    else:
        for val in statuses:
            try:
                intval = int(val)
            except (TypeError, ValueError):
                intval = 0
            if intval not in (0, 1, 2):
                intval = 2 if intval else 0
            cleaned.append(intval)
        if len(cleaned) < total:
            cleaned.extend([0] * (total - len(cleaned)))
        elif len(cleaned) > total:
            cleaned = cleaned[:total]
    return {
        "status": cleaned,
        "completed": [val == 2 for val in cleaned],
        "queue": None,
        "prompts": serialized,
    }


def _restrict_bpe_training_text(text: str) -> str:
    pieces: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if ch in ASCII_LETTERS:
            start = i
            i += 1
            while i < length and text[i] in ASCII_LETTERS:
                i += 1
            pieces.append(text[start:i])
            continue
        if ch == " ":
            pieces.append(ch)
            i += 1
            continue
        if ch in "\n\r\t":
            pieces.append(ch)
            i += 1
            continue
        pieces.append(" ")
        pieces.append(ch)
        pieces.append(" ")
        i += 1
    return "".join(pieces)


# -----------------------------------------------------------------------------
# Data Utilities
# -----------------------------------------------------------------------------


class GPT2TokenizerWrapper:
    EXTRA_SPECIAL_TOKENS = list(SPECIAL_TOKENS)

    def __init__(
        self,
        train_text: str,
        cache_path: pathlib.Path,
        vocab_size: int,
        pretrained_json: str | None = None,
    ) -> None:
        if not cache_path.parent.exists():
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        self.cache_path = cache_path
        self.pretrained_json = pretrained_json
        self.extra_special_tokens: list[str] = list(self.EXTRA_SPECIAL_TOKENS)
        self.tokenizer = self._load_or_train(
            train_text,
            cache_path,
            vocab_size,
            self.extra_special_tokens,
        )
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.non_special_ids = [
            tok_id for tok_id in range(self.vocab_size) if tok_id not in self.special_ids
        ]
        self.leading_alpha_token_ids = sorted(self._collect_leading_alpha_tokens())

    def _load_or_train(
        self,
        train_text: str,
        cache_path: pathlib.Path,
        vocab_size: int,
        extra_special_tokens: list[str],
    ) -> GPT2TokenizerFast:
        if cache_path.exists():
            return self._configure_special_tokens(
                GPT2TokenizerFast(tokenizer_file=str(cache_path)),
                extra_special_tokens,
            )
        if self.pretrained_json:
            tokenizer = Tokenizer.from_str(self.pretrained_json)
            tk = GPT2TokenizerFast(tokenizer_object=tokenizer)
            try:
                cache_path.write_text(self.pretrained_json, encoding="utf-8")
            except OSError:
                pass  # best-effort cache write
            return self._configure_special_tokens(tk, extra_special_tokens)
        tokenizer = Tokenizer(BPE(unk_token=None))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        byte_values = sorted(set(train_text.encode("utf-8")))
        initial_alphabet = [chr(b) for b in byte_values] or ByteLevel.alphabet()
        # Reserve one additional slot beyond the requested vocab size to compensate
        # for the underlying trainer implicitly injecting an end-of-input token.
        trainer_vocab_size = vocab_size + 1
        trainer = BpeTrainer(
            vocab_size=trainer_vocab_size,
            min_frequency=2,
            special_tokens=[],
            initial_alphabet=initial_alphabet,
        )
        sanitized = _restrict_bpe_training_text(train_text)
        tokenizer.train_from_iterator([sanitized], trainer=trainer)
        tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
        if extra_special_tokens:
            tokenizer.add_special_tokens(extra_special_tokens)
        tokenizer.save(str(cache_path))
        tk = GPT2TokenizerFast(tokenizer_file=str(cache_path))
        return self._configure_special_tokens(tk, extra_special_tokens)

    def _configure_special_tokens(
        self, tk: GPT2TokenizerFast, extra_special_tokens: list[str]
    ) -> GPT2TokenizerFast:
        # Suggested GPT-2 style special tokens (BOS/EOS/UNK/PAD) are omitted for now.
        if extra_special_tokens:
            tk.add_special_tokens({"additional_special_tokens": extra_special_tokens})
        return tk

    def _collect_leading_alpha_tokens(self) -> set[int]:
        token_ids: set[int] = set()
        for tok_id in self.non_special_ids:
            try:
                piece = self.tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
            except KeyError:
                continue
            if not piece:
                continue
            first = piece[0]
            if first in ASCII_LOWERCASE:
                token_ids.add(tok_id)
        return token_ids

    def encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(ids, dtype=torch.long)

    def encode_corpus(self, text: str, chunk_chars: int = 2048) -> torch.Tensor:
        ids: list[int] = []
        for i in range(0, len(text), chunk_chars):
            piece = text[i : i + chunk_chars]
            if not piece:
                continue
            ids.extend(self.tokenizer.encode(piece, add_special_tokens=False))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens.tolist())

    def decode_one(self, token: int) -> str:
        return self.tokenizer.decode([token])

    def decode_pretty(self, settings: Settings, tokens: torch.Tensor, color: str = Colors.MAGENTA, altcolor: str = Colors.GREEN, alt: bool = False) -> str:
        if alt: color, altcolor =  Colors.YELLOW, Colors.CYAN
        parts = []
        for tok in tokens.tolist():
            s = self.decode_one(tok)
            assert s, "got empty token"
            s = s.replace("\n", FANCY_ENTER if settings.escape_newline_tokens else FANCY_ENTER.replace(" ", "\n"))
            if not s.replace(" ", ""):
                s = s.replace(" ", FANCY_SPACE)
            parts.append(color + s + Colors.RESET)
            color, altcolor = altcolor, color
        return "".join(parts)


class PromptTracker:
    def __init__(self, tokenizer: GPT2TokenizerWrapper, state: dict | None = None) -> None:
        self.tokenizer = tokenizer
        self.prompts: list[tuple[str, str]] = []
        self.status: list[int] = []  # 0=unsolved, 1=solved via sampling, 2=solved via argmax
        self._cache: dict[int, torch.Tensor] = {}
        self._expected_token_ids: dict[int, List[int]] = {}
        self._queue: list[int] | None = None
        self.load_state(state)

    def load_state(self, state: dict | None) -> None:
        self._cache.clear()
        self._expected_token_ids.clear()
        prompts_state = state.get("prompts") if isinstance(state, dict) else None
        prompt_entries: list[tuple[str, str]] = []
        cleared_prompts = isinstance(prompts_state, list) and len(prompts_state) == 0
        if isinstance(prompts_state, list):
            for entry in prompts_state:
                prompt_text = None
                expected_text = None
                if isinstance(entry, dict):
                    prompt_text = entry.get("prompt")
                    expected_text = entry.get("expected")
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    prompt_text, expected_text = entry[0], entry[1]
                if isinstance(prompt_text, str) and isinstance(expected_text, str):
                    prompt_entries.append((prompt_text, expected_text))
        if prompt_entries:
            self.prompts = prompt_entries
        elif cleared_prompts:
            self.prompts = []
        else:
            self.prompts = default_prompt_entries()
        total = len(self.prompts)
        raw_status: list[int] | None = None
        if state and isinstance(state.get("status"), list):
            raw_status = state.get("status")
        elif state and isinstance(state.get("completed"), list):
            raw_status = [2 if bool(val) else 0 for val in state.get("completed", [])]
        if raw_status is None:
            cleaned = [0] * total
        else:
            cleaned = []
            for val in raw_status:
                try:
                    intval = int(val)
                except (TypeError, ValueError):
                    intval = 0
                if intval not in (0, 1, 2):
                    intval = 2 if intval else 0
                cleaned.append(intval)
        if len(cleaned) < total:
            cleaned.extend([0] * (total - len(cleaned)))
        elif len(cleaned) > total:
            cleaned = cleaned[:total]
        self.status = cleaned
        queue_state: list[int] | None = None
        if state and isinstance(state.get("queue"), list):
            queue_state = [
                idx
                for idx in (int(val) for val in state["queue"])
                if 0 <= idx < total and self.status[idx] < 2
            ]
        self._queue = queue_state if queue_state else None

    def serialize(self) -> dict:
        return {
            "status": list(self.status),
            "completed": [val == 2 for val in self.status],
            "queue": list(self._queue) if self._queue else None,
            "prompts": [
                {"prompt": prompt, "expected": expected} for prompt, expected in self.prompts
            ],
        }

    def next_goal(self) -> tuple[int | None, tuple[str, str] | None]:
        for idx, val in enumerate(self.status):
            if val < 2:
                return idx, self.prompts[idx]
        return None, None

    def prompt_tensor(self, idx: int, device: torch.device) -> torch.Tensor:
        if idx not in self._cache:
            tensor = self.tokenizer.encode(self.prompts[idx][0]).unsqueeze(0)
            self._cache[idx] = tensor
        return self._cache[idx].to(device)

    def expected_text(self, idx: int) -> str:
        return self.prompts[idx][1]
    
    def expected_token_ids(self, idx: int) -> List[int]:
        if idx not in self._expected_token_ids:
            tensor = self.tokenizer.encode(self.expected_text(idx))
            self._expected_token_ids[idx] = tensor.tolist()
        return self._expected_token_ids[idx]

    def state(self, idx: int) -> int:
        if idx < 0 or idx >= len(self.status):
            return 2
        return self.status[idx]

    def mark_if_satisfied(
        self, idx: int, completion_ids: List[int], *, used_argmax: bool
    ) -> tuple[bool, int, int]:
        if idx is None or idx < 0 or idx >= len(self.prompts):
            return False, 0, 0
        prev = self.status[idx]
        expected_ids = self.expected_token_ids(idx)
        if len(completion_ids) < len(expected_ids):
            return False, prev, prev
        matched = completion_ids[: len(expected_ids)] == expected_ids
        new_state = prev
        if matched:
            if used_argmax:
                new_state = 2
            elif prev == 0:
                new_state = 1
        self.status[idx] = new_state
        return matched, prev, new_state

    def remaining(self) -> int:
        return sum(1 for val in self.status if val < 2)

    def is_completed(self, idx: int | None) -> bool:
        if idx is None:
            return True
        if idx < 0 or idx >= len(self.status):
            return True
        return self.status[idx] == 2

    def pending_indices(self, limit: int | None = None) -> List[int]:
        indices = [i for i, val in enumerate(self.status) if val < 2]
        if limit is not None:
            return indices[: max(0, int(limit))]
        return indices

    def prompt_queue(self, *, reset: bool) -> list[int]:
        if reset or self._queue is None:
            self._queue = self.pending_indices()
        else:
            self._queue = [idx for idx in self._queue if self.status[idx] < 2]
            if not self._queue:
                self._queue = self.pending_indices()
        return self._queue

    def counts(self) -> tuple[int, int, int]:
        random_or_better = sum(1 for val in self.status if val > 0)
        solved = sum(1 for val in self.status if val == 2)
        total = len(self.status)
        return random_or_better, solved, total


@dataclass
class TextDataset:
    train_tokens: torch.Tensor
    test_tokens: torch.Tensor
    train_text: str | None
    test_text: str | None
    train_bytes: int
    test_bytes: int
    train_path: pathlib.Path
    test_path: pathlib.Path
    positions: Dict[str, int] = field(
        default_factory=lambda: {"train": 0, "test": 0}
    )
    byte_positions: Dict[str, int] = field(
        default_factory=lambda: {"train": 0, "test": 0}
    )
    chunks: Dict[str, torch.Tensor] = field(default_factory=dict)
    chunk_offsets: Dict[str, int] = field(default_factory=dict)

    def state_dict(self) -> Dict[str, Dict[str, int]]:
        return {
            "positions": dict(self.positions),
            "byte_positions": dict(self.byte_positions),
        }

    def load_state(self, state: Dict[str, Dict[str, int]]) -> None:
        self.positions.update(state.get("positions", {}))
        self.byte_positions.update(state.get("byte_positions", {}))
        for split in ("train", "test"):
            data = self.train_tokens if split == "train" else self.test_tokens
            total = len(data)
            if total:
                self.positions[split] %= total
            byte_total = self.train_bytes if split == "train" else self.test_bytes
            if byte_total:
                self.byte_positions[split] %= byte_total

    def prepare_cycle(self, split: str, total_chars: int) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split {split!r}")
        source = self.train_tokens if split == "train" else self.test_tokens
        text = self.train_text if split == "train" else self.test_text
        if total_chars <= 0 or total_chars > len(source):
            total_chars = len(source)
        start = self.positions[split]
        chunk, parts_text = self._slice_with_wrap(source, text, start, total_chars)
        if len(chunk) <= 1:
            raise ValueError(f"Not enough tokens in {split} split to build a chunk")
        self.positions[split] = (start + total_chars) % len(source)
        self._byte_segments(split, parts_text)
        self.chunks[split] = chunk
        self.chunk_offsets[split] = start % len(source)

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

    def _byte_segments(self, split: str, parts_text: list[str]) -> list[tuple[int, int]]:
        segments = []
        byte_pos = self.byte_positions[split]
        total_bytes = self.train_bytes if split == "train" else self.test_bytes
        if total_bytes == 0:
            return segments
        for idx, part in enumerate(parts_text):
            part_bytes = len(part.encode("utf-8"))
            if not part_bytes:
                continue
            if idx == 0:
                start = byte_pos
                end = start + part_bytes
                byte_pos = end % total_bytes
            else:
                start = 0
                end = part_bytes
                byte_pos = part_bytes % total_bytes
            if end > total_bytes and idx == 0:
                end = total_bytes
            segments.append((start, end))
        self.byte_positions[split] = byte_pos % total_bytes
        return segments




@dataclass
class SpecialRowMasks:
    context_special_rows: set[int]
    context_disabled_mask: torch.Tensor | None
    context_bias_disabled_mask: torch.Tensor | None
    xctx_disabled_mask: torch.Tensor | None
    xctx_bias_disabled_mask: torch.Tensor | None
    context_dropout_positions: torch.Tensor | None
    attention_disabled_mask: torch.Tensor | None
    attention_dropout_positions: torch.Tensor | None
    encode_rows: torch.Tensor | None
    recode_rows: torch.Tensor | None
    recode_boundaries: torch.Tensor | None


def build_special_row_masks(
    batch_size: int,
    block_size: int,
    device: torch.device,
    *,
    context_enabled: bool,
    xctx_enabled: bool,
) -> SpecialRowMasks:
    context_special_rows: set[int] = set()
    context_disabled_mask: torch.Tensor | None = None
    xctx_disabled_mask: torch.Tensor | None = None
    context_dropout_positions: torch.Tensor | None = None
    attention_disabled_mask: torch.Tensor | None = None
    attention_dropout_positions: torch.Tensor | None = None
    if batch_size <= 0:
        return SpecialRowMasks(
            context_special_rows,
            context_disabled_mask,
            None,
            xctx_disabled_mask,
            None,
            context_dropout_positions,
            attention_disabled_mask,
            attention_dropout_positions,
            None,
            None,
            None,
        )

    def pick_row(
        occupied: set[int], disallowed: set[int] | None = None
    ) -> int:
        blocked = set(occupied)
        if disallowed:
            blocked |= set(disallowed)
        available = [idx for idx in range(batch_size) if idx not in blocked]
        if not available:
            available = list(range(batch_size))
        choice = random.choice(available)
        occupied.add(choice)
        return choice

    occupied_rows: set[int] = set()
    if context_enabled:
        context_disabled_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        full_off = pick_row(occupied_rows)
        context_disabled_mask[full_off] = True
        context_special_rows.add(full_off)
    if xctx_enabled:
        xctx_disabled_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        context_dropout_positions = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
            device=device,
        )
        no_xctx_row = pick_row(occupied_rows)
        xctx_disabled_mask[no_xctx_row] = True
        context_special_rows.add(no_xctx_row)
        puncture_row = pick_row(occupied_rows)
        drop_position = random.randrange(max(1, block_size))
        context_dropout_positions[puncture_row] = drop_position
        context_special_rows.add(puncture_row)
        attention_disabled_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        attention_dropout_positions = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
            device=device,
        )
        att_off = pick_row(occupied_rows)
        attention_disabled_mask[att_off] = True
        context_special_rows.add(att_off)
        att_puncture = pick_row(occupied_rows)
        att_drop_position = random.randrange(max(1, block_size))
        attention_dropout_positions[att_puncture] = att_drop_position
        context_special_rows.add(att_puncture)

    return SpecialRowMasks(
        context_special_rows,
        context_disabled_mask,
        None,
        xctx_disabled_mask,
        None,
        context_dropout_positions,
        attention_disabled_mask,
        attention_dropout_positions,
        None,
        None,
        None,
    )


def load_or_prepare_tokens(
    split: str,
    text_path: str,
    text: str | None,
    cache_path: pathlib.Path,
    tokenizer: GPT2TokenizerWrapper,
    seed: int,
) -> Tuple[torch.Tensor, str | None, int, int]:
    if cache_path.exists():
        payload = torch.load(cache_path)
        tokens = payload["tokens"].long()
        bytes_count = int(payload.get("bytes", 0))
        inserts = int(payload.get("inserts", 0))
        trimmed_text = text
        print(color_text(f"Loaded cached {split} tokens from {cache_path}", Colors.YELLOW))
        return tokens, trimmed_text, bytes_count, inserts

    print(color_text(f"Tokenizing raw {split} data: {text_path}...", Colors.BLUE))

    if text is None:
        raise FileNotFoundError(
            f"No cached tokens at {cache_path} and source text missing for {split}."
        )
    trimmed_text = text
    if not trimmed_text:
        raise ValueError(f"Text for {split} split is empty")
    tokens = tokenizer.encode_corpus(trimmed_text)
    bytes_count = len(trimmed_text.encode("utf-8"))
    inserts = 0
    torch.save({"tokens": tokens, "bytes": bytes_count, "inserts": inserts}, cache_path)
    print(color_text(f"Saved {split} token cache to {cache_path}", Colors.YELLOW))
    return tokens, trimmed_text, bytes_count, inserts


# -----------------------------------------------------------------------------
# GRCE Model Components
# -----------------------------------------------------------------------------


def _linear_params(in_dim: int, out_dim: int) -> int:
    return in_dim * out_dim + out_dim


def _layernorm_params(dim: int) -> int:
    return 2 * dim


def _module_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _module_list_param_count(modules: nn.ModuleList) -> int:
    return sum(_module_param_count(m) for m in modules)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
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
    ) -> torch.Tensor:
        B, T, C = x.shape
        k_full = self.key(x)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v_full = self.value(x)
        atten_block_mask: torch.Tensor | None = None
        if dropout_positions is not None:
            valid = (dropout_positions >= 0).nonzero(as_tuple=False).flatten()
            if valid.numel() > 0:
                atten_block_mask = torch.zeros(B, T, T, dtype=torch.bool, device=x.device)
                for b_idx in valid.tolist():
                    pos = int(dropout_positions[b_idx].item())
                    if 0 <= pos < T - 1:
                        atten_block_mask[b_idx, pos + 1 :, pos] = True
        k = k_full.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v_full.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if not full_attention:
            att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        if atten_block_mask is not None:
            att = att.masked_fill(atten_block_mask[:, None, :, :], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        if disable_rows is not None and disable_rows.any():
            row_mask = (~disable_rows).view(-1, 1, 1).to(y.dtype)
            y = y * row_mask
        return self.proj(y)

    def forward_incremental(
        self,
        x: torch.Tensor,
        cache: "LayerCache",
        *,
        puncture_mask: torch.Tensor | None = None,
        disable_rows: torch.Tensor | None = None,
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
        cache.key = torch.cat([cache.key, key_append], dim=2)
        cache.value = torch.cat([cache.value, value_append], dim=2)
        cache.length = cache.key.size(2)
        k = cache.key
        v = cache.value
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
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
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
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        record_mask: bool = False,
        attention_disabled_rows: torch.Tensor | None = None,
        attention_dropout_positions: torch.Tensor | None = None,
        full_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out = self.attn(
            self.ln1(x),
            disable_rows=attention_disabled_rows,
            dropout_positions=attention_dropout_positions,
            full_attention=full_attention,
        )
        if attention_disabled_rows is not None and attention_disabled_rows.any():
            mask = (~attention_disabled_rows).view(-1, 1, 1).to(attn_out.dtype)
            attn_out = attn_out * mask
        x = x + attn_out
        pre_ff = self.ln2(x)
        ff_out, mask = self.ff(pre_ff, record_mask=record_mask)
        x = x + ff_out
        return x, mask

    def forward_incremental(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        *,
        record_mask: bool = False,
        attention_disabled_rows: torch.Tensor | None = None,
        puncture_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LayerCache, torch.Tensor | None]:
        attn_out, cache = self.attn.forward_incremental(
            self.ln1(x),
            cache,
            puncture_mask=puncture_mask,
            disable_rows=attention_disabled_rows,
        )
        if attention_disabled_rows is not None and attention_disabled_rows.any():
            mask = (~attention_disabled_rows).view(-1, 1, 1).to(attn_out.dtype)
            attn_out = attn_out * mask
        x = x + attn_out
        pre_ff = self.ln2(x)
        ff_out, mask = self.ff(pre_ff, record_mask=record_mask)
        x = x + ff_out
        return x, cache, mask


@dataclass
class LayerCache:
    key: torch.Tensor
    value: torch.Tensor
    length: int = 0


class GPTCore(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.detach_layer = max(-1, int(getattr(config, "detach_layer", -1)))
        if self.detach_layer > len(self.blocks):
            self.detach_layer = len(self.blocks)

    def allocate_kv_caches(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
    ) -> list[LayerCache]:
        head_dim = self.config.n_embd // self.config.n_head
        dtype = self.tok_emb.weight.dtype
        caches: list[LayerCache] = []
        for _ in range(len(self.blocks)):
            key = torch.empty(
                batch_size,
                self.config.n_head,
                0,
                head_dim,
                device=device,
                dtype=dtype,
            )
            value = torch.empty_like(key)
            caches.append(LayerCache(key=key, value=value, length=0))
        return caches

    def _build_puncture_mask(
        self,
        positions: torch.Tensor | None,
        cache_len: int,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        if positions is None or cache_len <= 1:
            return None
        valid = (positions >= 0) & (positions < (cache_len - 1))
        if not torch.any(valid):
            return None
        mask = torch.zeros(positions.size(0), cache_len, dtype=torch.bool, device=device)
        mask[valid, positions[valid]] = True
        return mask

    def forward(
        self,
        idx: torch.Tensor,
        block_biases: List[torch.Tensor] | None = None,
        *,
        record_relu_mask: bool = False,
        attention_disabled_rows: torch.Tensor | None = None,
        attention_dropout_positions: torch.Tensor | None = None,
        full_attention: bool = False,
        target_position: int | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor] | None]:
        B, T = idx.shape
        device = idx.device
        tok = self.tok_emb(idx)
        if position_ids is None:
            base_positions = torch.arange(T, device=device).unsqueeze(0)
            pos_idx = base_positions.expand(B, -1)
        else:
            if position_ids.dim() == 1:
                pos_idx = position_ids.view(B, T)
            else:
                pos_idx = position_ids
        if torch.any(pos_idx >= self.config.block_size):
            raise ValueError("position ids exceed configured --block-size")
        pos = self.pos_emb(pos_idx)
        x = self.drop(tok + pos)
        if attention_disabled_rows is not None:
            attention_disabled_rows = attention_disabled_rows.to(device=device, dtype=torch.bool)
        else:
            attention_disabled_rows = torch.zeros(B, dtype=torch.bool, device=device)
        if attention_dropout_positions is not None:
            attention_dropout_positions = attention_dropout_positions.to(device=device, dtype=torch.long)
        block_inputs: List[torch.Tensor] = []
        relu_masks: List[torch.Tensor | None] | None = None
        if record_relu_mask:
            relu_masks = [None] * len(self.blocks)
        target_idx = target_position if target_position is not None else (T - 1)
        target_idx = max(0, min(T - 1, int(target_idx)))
        for layer_idx, block in enumerate(self.blocks):
            if block_biases is not None:
                x = x + block_biases[layer_idx]
            block_inputs.append(x[:, target_idx, :])
            x, layer_mask = block(
                x,
                record_mask=record_relu_mask,
                attention_disabled_rows=attention_disabled_rows,
                attention_dropout_positions=attention_dropout_positions,
                full_attention=full_attention,
            )
            if self.detach_layer > 0 and (layer_idx + 1) == self.detach_layer:
                x = x.detach()
            if record_relu_mask and relu_masks is not None and layer_mask is not None:
                relu_masks[layer_idx] = layer_mask
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, x, block_inputs, relu_masks

    def forward_step(
        self,
        idx: torch.Tensor,
        position_ids: torch.Tensor,
        caches: list[LayerCache],
        block_biases: list[torch.Tensor] | None = None,
        *,
        record_relu_mask: bool = False,
        attention_disabled_rows: torch.Tensor | None = None,
        puncture_positions: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor] | None]:
        if attention_disabled_rows is not None:
            attention_disabled_rows = attention_disabled_rows.to(idx.device, dtype=torch.bool)
        else:
            attention_disabled_rows = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        pos_idx = position_ids.view(idx.size(0), 1)
        tok = self.tok_emb(idx.view(idx.size(0), 1))
        pos = self.pos_emb(pos_idx)
        x = self.drop(tok + pos)
        block_inputs: List[torch.Tensor] = []
        relu_masks: List[torch.Tensor | None] | None = None
        if record_relu_mask:
            relu_masks = [None] * len(self.blocks)
        cache_len = caches[0].length
        puncture_mask = self._build_puncture_mask(
            puncture_positions,
            cache_len + 1,
            device=idx.device,
        )
        for layer_idx, block in enumerate(self.blocks):
            if block_biases is not None:
                x = x + block_biases[layer_idx].unsqueeze(1)
            block_inputs.append(x[:, 0, :])
            x, caches[layer_idx], layer_mask = block.forward_incremental(
                x,
                caches[layer_idx],
                record_mask=record_relu_mask,
                attention_disabled_rows=attention_disabled_rows,
                puncture_mask=puncture_mask,
            )
            if self.detach_layer > 0 and (layer_idx + 1) == self.detach_layer:
                x = x.detach()
            if record_relu_mask and relu_masks is not None and layer_mask is not None:
                relu_masks[layer_idx] = layer_mask
        x = self.ln_f(x)
        logits = self.head(x)
        return logits[:, 0, :], x[:, 0, :], block_inputs, relu_masks


class BaseContextChannel(nn.Module):
    def __init__(self, config: ModelConfig, width: int) -> None:
        super().__init__()
        self.config = config
        self.context_dim = int(width)
        self.disabled = self.context_dim <= 0
        self.detach_span = max(0, int(getattr(config, "detach_span", 0)))
        self.detach_context = bool(getattr(config, "detach_context", True))

    def parameter_breakdown(self) -> dict[str, int]:
        return {}

    @staticmethod
    def _apply_mask(tensor: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return tensor
        masked = tensor.clone()
        masked[mask] = 0
        return masked

    def normalize_state(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class GRCEContextChannel(BaseContextChannel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config, config.n_grce)
        self.is_xctx = False
        if self.disabled:
            return
        self.sample_norms = nn.ModuleList(
            nn.LayerNorm(config.n_embd) for _ in range(config.n_layer)
        )
        self.sample_projections = nn.ModuleList(
            nn.Linear(config.n_embd, self.context_dim) for _ in range(config.n_layer)
        )
        self.bias_norm = nn.LayerNorm(self.context_dim)
        self.bias_projections = nn.ModuleList(
            nn.Linear(self.context_dim, config.n_embd) for _ in range(config.n_layer)
        )
        self.mix_norm = nn.LayerNorm(self.context_dim)
        hidden = max(1, 4 * self.context_dim)
        self.mlp_up = nn.Linear(self.context_dim, hidden)
        self.mlp_down = nn.Linear(hidden, self.context_dim)
        self.output_norm = nn.LayerNorm(self.context_dim)

    def project(self, context: torch.Tensor) -> List[torch.Tensor]:
        if self.disabled:
            raise RuntimeError("Context channel disabled; project should not be called.")
        normed = self.bias_norm(context)
        return [proj(normed) for proj in self.bias_projections]

    def update(
        self,
        block_inputs: List[torch.Tensor],
        *,
        prev_context: torch.Tensor | None,
        stop_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.disabled:
            raise RuntimeError("Context channel disabled; update should not be called.")
        messages: list[torch.Tensor] = []
        for norm, proj, part in zip(self.sample_norms, self.sample_projections, block_inputs):
            messages.append(proj(norm(part)))
        fused = torch.stack(messages, dim=0).sum(dim=0)
        if prev_context is None:
            combined = fused
        else:
            residual = prev_context.detach() if (stop_grad and self.detach_context) else prev_context
            combined = fused + residual
        mixed = self.mix_norm(combined)
        mlp_out = self.mlp_down(F.relu(self.mlp_up(mixed)))
        raw_context = mixed + mlp_out
        context = self.output_norm(raw_context)
        return context, raw_context

    def parameter_breakdown(self) -> dict[str, int]:
        if self.disabled:
            return {}
        sampler = _module_list_param_count(self.sample_norms) + _module_list_param_count(
            self.sample_projections
        )
        mlp = (
            _module_param_count(self.mix_norm)
            + _module_param_count(self.mlp_up)
            + _module_param_count(self.mlp_down)
            + _module_param_count(self.output_norm)
        )
        bias = _module_param_count(self.bias_norm) + _module_list_param_count(self.bias_projections)
        return {"samplers": sampler, "mlp": mlp, "bias": bias}

    def normalize_state(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            return tensor
        return self.output_norm(tensor)


class XCTXContextChannel(BaseContextChannel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config, config.n_xctx)
        self.is_xctx = True
        if self.disabled:
            return
        layers = max(1, config.n_layer)
        mid = max(1, min(config.n_embd // 2, self.context_dim // 2, max(config.n_embd // 4, self.context_dim // layers)))
        down_dim = max(1, self.context_dim // 2)
        self.inner_dim = mid
        self.down_dim = down_dim
        self.sample_input_norms = nn.ModuleList(
            nn.LayerNorm(config.n_embd) for _ in range(config.n_layer)
        )
        self.sample_e2u = nn.ModuleList(
            nn.Linear(config.n_embd, mid) for _ in range(config.n_layer)
        )
        self.sample_mid_norms = nn.ModuleList(
            nn.LayerNorm(mid) for _ in range(config.n_layer)
        )
        self.sample_u2x = nn.ModuleList(
            nn.Linear(mid, self.context_dim) for _ in range(config.n_layer)
        )
        self.bias_x2u = nn.ModuleList(
            nn.Linear(self.context_dim, mid) for _ in range(config.n_layer)
        )
        self.bias_mid_norms = nn.ModuleList(
            nn.LayerNorm(mid) for _ in range(config.n_layer)
        )
        self.bias_u2e = nn.ModuleList(
            nn.Linear(mid, config.n_embd) for _ in range(config.n_layer)
        )
        self.down_proj = nn.Linear(self.context_dim, down_dim)
        self.mix_norm = nn.LayerNorm(down_dim)
        self.up_proj = nn.Linear(down_dim, self.context_dim)
        self.cross_proj = nn.Linear(self.context_dim, self.context_dim)
        self.rms_norm = RMSNorm(self.context_dim)

    def project(self, context: torch.Tensor) -> List[torch.Tensor]:
        if self.disabled:
            raise RuntimeError("Context channel disabled; project should not be called.")
        outputs: list[torch.Tensor] = []
        for x2u, norm, u2e in zip(self.bias_x2u, self.bias_mid_norms, self.bias_u2e):
            reduced = norm(x2u(context))
            outputs.append(u2e(reduced))
        return outputs

    def update(
        self,
        block_inputs: List[torch.Tensor],
        *,
        prev_context: torch.Tensor | None,
        stop_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.disabled:
            raise RuntimeError("Context channel disabled; update should not be called.")
        messages: list[torch.Tensor] = []
        for inp_norm, e2u, mid_norm, u2x, part in zip(
            self.sample_input_norms,
            self.sample_e2u,
            self.sample_mid_norms,
            self.sample_u2x,
            block_inputs,
        ):
            reduced = e2u(inp_norm(part))
            messages.append(u2x(mid_norm(reduced)))
        stacked = torch.stack(messages, dim=0).sum(dim=0)
        if prev_context is None:
            base = torch.zeros_like(stacked)
        else:
            base = prev_context.detach() if (stop_grad and self.detach_context) else prev_context
        combined = base + stacked
        compressed = self.down_proj(combined)
        mixed = self.mix_norm(compressed)
        expanded = self.up_proj(mixed)
        remixed = self.cross_proj(F.relu(expanded))
        raw_context = base + remixed
        context = self.rms_norm(raw_context)
        return context, raw_context

    def parameter_breakdown(self) -> dict[str, int]:
        if self.disabled:
            return {}
        sampler = (
            _module_list_param_count(self.sample_input_norms)
            + _module_list_param_count(self.sample_e2u)
            + _module_list_param_count(self.sample_mid_norms)
            + _module_list_param_count(self.sample_u2x)
        )
        bias = (
            _module_list_param_count(self.bias_x2u)
            + _module_list_param_count(self.bias_mid_norms)
            + _module_list_param_count(self.bias_u2e)
        )
        mlp = (
            _module_param_count(self.down_proj)
            + _module_param_count(self.mix_norm)
            + _module_param_count(self.up_proj)
            + _module_param_count(self.cross_proj)
            + _module_param_count(self.rms_norm)
        )
        return {"samplers": sampler, "mlp": mlp, "bias": bias}

    def normalize_state(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            return tensor
        return self.rms_norm(tensor)


class GRCEGPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.core = GPTCore(config)
        self.context_channels = nn.ModuleList()
        if config.n_grce > 0:
            self.context_channels.append(
                GRCEContextChannel(config)
            )
        if config.n_xctx > 0:
            self.context_channels.append(
                XCTXContextChannel(config)
            )

    def forward_autoreg(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        disable_context: bool = False,
        capture_activations: bool = False,
        collect_relu_mask: bool = False,
        context_disabled_rows: torch.Tensor | None = None,
        context_bias_disabled_rows: torch.Tensor | None = None,
        xctx_disabled_rows: torch.Tensor | None = None,
        xctx_bias_disabled_rows: torch.Tensor | None = None,
        context_dropout_positions: torch.Tensor | None = None,
        disable_xctx: bool = False,
        attention_disabled_rows: torch.Tensor | None = None,
        attention_dropout_positions: torch.Tensor | None = None,
        initial_context_raw: list[torch.Tensor] | None = None,
        record_final_context: bool = False,
        encoder_mode: bool = False,
        position_offsets: torch.Tensor | None = None,
        encode_rows: torch.Tensor | None = None,
        recode_rows: torch.Tensor | None = None,
        recode_boundaries: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        B, T = idx.shape
        device = idx.device
        if position_offsets is None:
            position_offsets = torch.zeros(B, dtype=torch.long, device=device)
        else:
            position_offsets = position_offsets.to(device=device, dtype=torch.long)
            if position_offsets.dim() != 1 or position_offsets.shape[0] != B:
                raise ValueError("position_offsets must be 1D with batch_size entries")
        active_channels: list[GRCEContextChannel] = []
        active_indices: list[int] = []
        if not disable_context:
            for ch_idx, channel in enumerate(self.context_channels):
                if channel.is_xctx and disable_xctx:
                    continue
                if channel.disabled:
                    continue
                active_channels.append(channel)
                active_indices.append(ch_idx)
        use_context = bool(active_channels)
        if context_disabled_rows is not None:
            context_disabled_rows = context_disabled_rows.to(device=device, dtype=torch.bool)
        if context_disabled_rows is None or not use_context:
            context_disabled_rows = torch.zeros(B, dtype=torch.bool, device=device)
        if context_bias_disabled_rows is not None:
            context_bias_disabled_rows = context_bias_disabled_rows.to(device=device, dtype=torch.bool)
        if not use_context:
            context_bias_disabled_rows = None
        if xctx_disabled_rows is not None:
            xctx_disabled_rows = xctx_disabled_rows.to(device=device, dtype=torch.bool)
        if xctx_disabled_rows is None or not use_context:
            xctx_disabled_rows = torch.zeros(B, dtype=torch.bool, device=device)
        if xctx_bias_disabled_rows is not None:
            xctx_bias_disabled_rows = xctx_bias_disabled_rows.to(device=device, dtype=torch.bool)
        if not use_context:
            xctx_bias_disabled_rows = None
        if context_dropout_positions is not None:
            context_dropout_positions = context_dropout_positions.to(device=device, dtype=torch.long).clone()
        if attention_disabled_rows is not None:
            attention_disabled_rows = attention_disabled_rows.to(device=device, dtype=torch.bool)
        else:
            attention_disabled_rows = torch.zeros(B, dtype=torch.bool, device=device)
        if attention_dropout_positions is not None:
            attention_dropout_positions = attention_dropout_positions.to(device=device, dtype=torch.long).clone()
        if encode_rows is not None:
            encode_rows = encode_rows.to(device=device, dtype=torch.bool)
        if recode_rows is not None:
            recode_rows = recode_rows.to(device=device, dtype=torch.bool)
        if recode_boundaries is not None:
            recode_boundaries = recode_boundaries.to(device=device, dtype=torch.long)
        def ensure_dynamic_mask(mask: torch.Tensor | None) -> torch.Tensor:
            if mask is None:
                return torch.zeros(B, dtype=torch.bool, device=device)
            return mask
        base_context_bias_mask = context_bias_disabled_rows
        base_xctx_bias_mask = xctx_bias_disabled_rows
        context_states: list[torch.Tensor] = []
        effective_initial_context: list[torch.Tensor] | None = None
        if use_context:
            for channel in active_channels:
                context_states.append(torch.zeros(B, channel.context_dim, device=device))
            if initial_context_raw is not None:
                source = initial_context_raw
                if len(source) == len(self.context_channels) and active_indices:
                    source = [source[i] for i in active_indices]
                if len(source) != len(active_channels):
                    raise ValueError(
                        "initial_context_raw must match number of active context channels"
                    )
                effective_initial_context = source
                for idx_ch, (channel, init_raw) in enumerate(
                    zip(active_channels, effective_initial_context)
                ):
                    if init_raw is None:
                        continue
                    if init_raw.device != device:
                        init_raw = init_raw.to(device)
                    expected = (B, channel.context_dim)
                    if init_raw.shape != expected:
                        raise ValueError(
                            f"initial context shape {init_raw.shape} does not match {expected}"
                        )
                    context_states[idx_ch] = channel.normalize_state(init_raw)
        logits_steps = []
        hidden_steps = []
        activation_store: dict | None = None
        need_store = capture_activations or collect_relu_mask
        relu_activity: list[list[torch.Tensor | None]] | None = [] if collect_relu_mask else None
        final_context_raw: list[torch.Tensor | None] | None = (
            [None for _ in active_channels] if (record_final_context and use_context) else None
        )
        if capture_activations:
            activation_store = {
                "block_norms": [[] for _ in range(self.config.n_layer)],
                "context_norms": [],
            }
        base_context_mask = context_disabled_rows
        base_context_mask_has = bool(base_context_mask.any().item())
        base_xctx_mask = xctx_disabled_rows
        base_xctx_mask_has = bool(base_xctx_mask.any().item())
        caches = None
        if not encoder_mode:
            caches = self.core.allocate_kv_caches(B, T, device=device)

        for t in range(T):
            context_bias_mask = (
                base_context_bias_mask.clone() if base_context_bias_mask is not None else None
            )
            xctx_bias_mask = (
                base_xctx_bias_mask.clone() if base_xctx_bias_mask is not None else None
            )
            xctx_step_mask = None
            xctx_mask_has = False
            if context_dropout_positions is not None:
                step_mask = context_dropout_positions == t
                if step_mask.any():
                    context_dropout_positions[step_mask] = -1
                    xctx_step_mask = step_mask
                    xctx_mask_has = True
            recode_encode_mask = None
            if recode_rows is not None and recode_boundaries is not None:
                recode_encode_mask = recode_rows & (recode_boundaries > t)
                if recode_encode_mask.any():
                    context_bias_mask = ensure_dynamic_mask(context_bias_mask)
                    context_bias_mask[recode_encode_mask] = True
                    xctx_bias_mask = ensure_dynamic_mask(xctx_bias_mask)
                    xctx_bias_mask[recode_encode_mask] = True
            if encoder_mode:
                prefix = idx
                target_pos = min(prefix.size(1) - 1, t)
                pos_seq = torch.arange(prefix.size(1), device=device, dtype=torch.long)
                position_matrix = position_offsets.view(B, 1) + pos_seq.view(1, -1)
            block_biases = None
            if use_context:
                for channel, state in zip(active_channels, context_states):
                    state_mask = base_context_mask
                    state_mask_has = base_context_mask_has
                    bias_mask = context_bias_mask
                    if channel.is_xctx:
                        state_mask = base_xctx_mask
                        state_mask_has = base_xctx_mask_has
                        bias_mask = xctx_bias_mask
                        if xctx_mask_has:
                            state_mask = (
                                (state_mask | xctx_step_mask)
                                if state_mask_has
                                else xctx_step_mask
                            )
                            state_mask_has = True
                    state_for_bias = state
                    if state_mask_has:
                        state_for_bias = state_for_bias.clone()
                        state_for_bias[state_mask] = 0
                    bias_vectors = channel.project(state_for_bias)
                    bias_mask_has = bool(bias_mask is not None and bias_mask.any())
                    channel_biases: list[torch.Tensor] = []
                    if encoder_mode:
                        for bias_vec in bias_vectors:
                            if state_mask_has:
                                bias_vec = bias_vec.clone()
                                bias_vec[state_mask] = 0
                            if bias_mask_has:
                                bias_vec = bias_vec.clone()
                                bias_vec[bias_mask] = 0
                            full = torch.zeros(
                                B,
                                prefix.size(1),
                                self.config.n_embd,
                                device=device,
                                dtype=bias_vec.dtype,
                            )
                            full[:, target_pos, :] = bias_vec
                            channel_biases.append(full)
                    else:
                        for bias_vec in bias_vectors:
                            working = bias_vec
                            if state_mask_has:
                                working = working.clone()
                                working[state_mask] = 0
                            if bias_mask_has:
                                working = working.clone()
                                working[bias_mask] = 0
                            channel_biases.append(working)
                    if block_biases is None:
                        block_biases = channel_biases
                    else:
                        for layer_idx in range(len(block_biases)):
                            block_biases[layer_idx] = (
                                block_biases[layer_idx] + channel_biases[layer_idx]
                            )
            if encoder_mode:
                logits, hidden_layer, block_inputs, layer_masks = self.core(
                    prefix,
                    block_biases=block_biases,
                    record_relu_mask=collect_relu_mask,
                    attention_disabled_rows=attention_disabled_rows,
                    attention_dropout_positions=attention_dropout_positions,
                    full_attention=encoder_mode,
                    target_position=target_pos,
                    position_ids=position_matrix,
                )
                step_logits = logits[:, target_pos : target_pos + 1, :]
                step_hidden = hidden_layer[:, target_pos : target_pos + 1, :]
            else:
                if caches is None:
                    raise RuntimeError("KV caches were not initialized")
                position_ids = position_offsets + t
                logits_step, hidden_step, block_inputs, layer_masks = self.core.forward_step(
                    idx[:, t],
                    position_ids,
                    caches,
                    block_biases=block_biases,
                    record_relu_mask=collect_relu_mask,
                    attention_disabled_rows=attention_disabled_rows,
                    puncture_positions=attention_dropout_positions,
                )
                step_logits = logits_step.unsqueeze(1)
                step_hidden = hidden_step.unsqueeze(1)
            if relu_activity is not None:
                relu_activity.append(layer_masks)
            if activation_store is not None:
                for layer_idx, block_inp in enumerate(block_inputs):
                    norms = torch.linalg.vector_norm(block_inp.detach(), dim=-1)
                    activation_store["block_norms"][layer_idx].extend(
                        norms.cpu().tolist()
                    )
            logits_steps.append(step_logits)
            hidden_steps.append(step_hidden)
            if use_context:
                for idx_ch, channel in enumerate(active_channels):
                    span = channel.detach_span
                    if span <= 0:
                        stop_grad = False
                    elif span == 1:
                        stop_grad = True
                    else:
                        stop_grad = (t % span == 0)
                    prev_context = context_states[idx_ch]
                    channel_mask = base_context_mask
                    channel_mask_has = base_context_mask_has
                    if channel.is_xctx:
                        if base_xctx_mask_has:
                            channel_mask = (
                                (channel_mask | base_xctx_mask)
                                if channel_mask_has
                                else base_xctx_mask
                            )
                            channel_mask_has = True
                        if xctx_mask_has:
                            channel_mask = (
                                (channel_mask | xctx_step_mask)
                                if channel_mask_has
                                else xctx_step_mask
                            )
                            channel_mask_has = True
                    if channel_mask_has:
                        prev_context = prev_context.clone()
                        prev_context[channel_mask] = 0
                    new_state, raw_context = channel.update(
                        block_inputs,
                        prev_context=prev_context,
                        stop_grad=stop_grad,
                    )
                    if channel_mask_has:
                        new_state = new_state.clone()
                        new_state[channel_mask] = 0
                        raw_context = raw_context.clone()
                        raw_context[channel_mask] = 0
                    context_states[idx_ch] = new_state
                    if final_context_raw is not None:
                        final_context_raw[idx_ch] = raw_context.detach()
                    if activation_store is not None:
                        ctx_norms = torch.linalg.vector_norm(raw_context.detach(), dim=-1)
                        activation_store["context_norms"].extend(ctx_norms.cpu().tolist())
        logits = torch.cat(logits_steps, dim=1)
        hidden = torch.cat(hidden_steps, dim=1)
        if relu_activity is not None:
            if activation_store is None and need_store:
                activation_store = {}
            if activation_store is not None:
                activation_store.setdefault("relu_activity", relu_activity)
        if final_context_raw is not None:
            if activation_store is None:
                activation_store = {}
            activation_store["final_context_raw"] = [
                ctx.clone() if ctx is not None else None for ctx in final_context_raw
            ]
        return logits, hidden, activation_store


def build_model_tag(config: ModelConfig) -> str:
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


ROW_METRIC_MAP = {
    "normal": "normal",
    "decode": "decode",
    "noxctx": "noxctx",
    "puxctx": "puxctx",
    "noattn": "noattn",
    "puattn": "puattn",
    "encode": "encode",
    "recode": "recode",
}

ROW_METRIC_HIST_KEYS = [
    "normal",
    "encode",
    "recode",
    "decode",
    "noxctx",
    "noattn",
    "puxctx",
    "puattn",
]

ROW_METRIC_LOG_KEYS = [
    "normal",
    #"encode",
    #"recode",
    "decode",
    "noxctx",
    "noattn",
    #"puxctx",
    #"puattn",
]

ROW_METRIC_LOG_GROUP = {
    "normal",
    "noxctx",
}


def sample_position_offsets(
    batch_size: int,
    block_size: int,
    block_length: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return per-row positional offsets when block_length < block_size."""

    if batch_size <= 0:
        return None
    if block_length >= block_size:
        return None
    max_offset = block_size - block_length
    if max_offset <= 0:
        return None
    return torch.randint(0, max_offset + 1, (batch_size,), device=device)


def aggregate_row_metrics(
    row_types: Sequence[str],
    per_token_losses: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float | None]:
    loss_sum = float((per_token_losses * valid_mask).sum().item())
    token_count = int(valid_mask.sum().item())
    metrics: dict[str, float | None] = {"target": loss_sum / max(1, token_count)}

    bucket_sum: dict[str, float] = defaultdict(float)
    bucket_count: dict[str, int] = defaultdict(int)
    for idx, row_type in enumerate(row_types):
        mask = valid_mask[idx]
        count = int(mask.sum().item())
        if count <= 0:
            continue
        bucket_sum[row_type] += float((per_token_losses[idx] * mask).sum().item())
        bucket_count[row_type] += count

    for row_type, metric_name in ROW_METRIC_MAP.items():
        denom = bucket_count.get(row_type, 0)
        if denom:
            metrics[metric_name] = bucket_sum[row_type] / denom
        else:
            metrics[metric_name] = None
    return metrics


def evaluate_single_batch(
    model: GRCEGPT,
    dataset: TextDataset,
    split: str,
    block_length: int,
    batch_size: int,
    device: torch.device,
    row_type_template: Sequence[str],
    *,
    context_enabled: bool,
    xctx_enabled: bool,
) -> dict[str, float | None]:
    row_types = list(row_type_template)
    random.shuffle(row_types)
    xb, yb = dataset.get_batch(split, block_length, batch_size, device)
    special_masks = build_row_type_masks(
        row_types,
        block_length,
        device,
        context_enabled=context_enabled,
        xctx_enabled=xctx_enabled,
    )
    position_offsets = sample_position_offsets(
        batch_size,
        model.config.block_size,
        block_length,
        device,
    )
    logits, _, _ = model.forward_autoreg(
        xb,
        targets=yb,
        context_disabled_rows=special_masks.context_disabled_mask,
        context_bias_disabled_rows=special_masks.context_bias_disabled_mask,
        xctx_disabled_rows=special_masks.xctx_disabled_mask,
        xctx_bias_disabled_rows=special_masks.xctx_bias_disabled_mask,
        context_dropout_positions=special_masks.context_dropout_positions,
        attention_disabled_rows=special_masks.attention_disabled_mask,
        attention_dropout_positions=special_masks.attention_dropout_positions,
        position_offsets=position_offsets,
        encode_rows=special_masks.encode_rows,
        recode_rows=special_masks.recode_rows,
        recode_boundaries=special_masks.recode_boundaries,
    )
    per_token = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        yb.view(-1),
        reduction="none",
        ignore_index=LOSS_IGNORE_INDEX,
    )
    per_token = per_token.view(batch_size, -1)
    valid_mask = (yb != LOSS_IGNORE_INDEX)
    return aggregate_row_metrics(row_types, per_token, valid_mask)

def train_model(
    settings: Settings,
    model: GRCEGPT,
    dataset: TextDataset,
    device: torch.device,
    steps: int,
    block_length: int,
    batch_size: int,
    eval_interval: int,
    start_step: int,
    sample_prompt: torch.Tensor,
    sample_chars: int,
    tokenizer: GPT2TokenizerWrapper,
    suppress_newlines: bool,
    newline_token_id: int | None,
    prompt_tracker: PromptTracker | None = None,
    reset_prompt_queue: bool = False,
    *,
    cycle_wall_start: float,
    base_wall_seconds: float,
    show_time: bool = False,
    default_prompt_boundary: bool = False,
    boundary_blocklist: Sequence[int] | None = None,
    show_train_loss_details: bool = False,
    show_test_loss_details: bool = True,
) -> Tuple[int, List[Dict[str, float]], float, float, float, float]:
    """Run the main training loop for a cycle."""

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    total_steps = start_step
    history_updates: List[Dict[str, float]] = []
    printed_header = False
    eval_interval = max(1, int(eval_interval))
    if prompt_tracker is not None:
        prompt_queue = prompt_tracker.prompt_queue(reset=reset_prompt_queue)
    else:
        prompt_queue = []

    context_path_enabled = bool(model.context_channels)
    xctx_enabled = any(
        getattr(channel, "is_xctx", False) for channel in getattr(model, "context_channels", [])
    )
    row_counts = compute_row_type_counts(batch_size)
    row_type_template = build_row_type_template(
        row_counts,
        batch_size,
        context_enabled=context_path_enabled,
        xctx_enabled=xctx_enabled,
    )

    loop_wall_start = time.time()
    loop_cpu_start = time.process_time()
    eval_wall_total = 0.0
    eval_cpu_total = 0.0

    long_loss_header = " ".join([""] + [f"{': ' if key in ROW_METRIC_LOG_GROUP else ''}{key}" for key in ROW_METRIC_LOG_KEYS])

    line_parts: List[str] = []
    if show_time:
        line_parts.append(color_text("time", Colors.BLUE))
    line_parts.append(color_text(f"step", Colors.CYAN))
    line_parts.append(color_text("train" + (long_loss_header if show_train_loss_details else ""), Colors.MAGENTA))
    line_parts.append(color_text("test" + (long_loss_header if show_test_loss_details else ""), Colors.GREEN))
    line = " | ".join(line_parts) + " |"
    print(line)

    for step in range(1, steps + 1):
        xb, yb = dataset.get_batch("train", block_length, batch_size, device)
        row_types = list(row_type_template)
        random.shuffle(row_types)
        special_masks = build_row_type_masks(
            row_types,
            block_length,
            device,
            context_enabled=context_path_enabled,
            xctx_enabled=xctx_enabled,
        )
        position_offsets = sample_position_offsets(
            batch_size,
            model.config.block_size,
            block_length,
            device,
        )
        logits, _, _ = model.forward_autoreg(
            xb,
            targets=yb,
            context_disabled_rows=special_masks.context_disabled_mask,
            context_bias_disabled_rows=special_masks.context_bias_disabled_mask,
            xctx_disabled_rows=special_masks.xctx_disabled_mask,
            xctx_bias_disabled_rows=special_masks.xctx_bias_disabled_mask,
            context_dropout_positions=special_masks.context_dropout_positions,
            attention_disabled_rows=special_masks.attention_disabled_mask,
            attention_dropout_positions=special_masks.attention_dropout_positions,
            position_offsets=position_offsets,
            encode_rows=special_masks.encode_rows,
            recode_rows=special_masks.recode_rows,
            recode_boundaries=special_masks.recode_boundaries,
        )
        total_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.view(-1),
            ignore_index=LOSS_IGNORE_INDEX,
        )

        optim.zero_grad(set_to_none=True)
        total_loss.backward()
        optim.step()
        total_steps += 1

        eval_due = step == 1 or step % eval_interval == 0 or step == steps
        if not eval_due:
            continue
        eval_wall_block = time.time()
        eval_cpu_block = time.process_time()
        model.eval()
        split_metrics: dict[str, dict[str, float | None]] = {}
        with torch.no_grad():
            for split in ("train", "test"):
                split_metrics[split] = evaluate_single_batch(
                    model,
                    dataset,
                    split,
                    block_length,
                    batch_size,
                    device,
                    row_type_template,
                    context_enabled=context_path_enabled,
                    xctx_enabled=xctx_enabled,
                )
        model.train()
        block_wall = time.time() - eval_wall_block
        block_cpu = time.process_time() - eval_cpu_block
        eval_wall_total += block_wall
        eval_cpu_total += block_cpu

        prompt_input = sample_prompt
        prompt_needs_boundary_flag = default_prompt_boundary and boundary_blocklist is not None
        current_prompt_idx = None
        use_argmax_completion = random.random() < 0.5
        sampling_strategy = "argmax" if use_argmax_completion else "sample"
        if prompt_tracker is not None:
            while prompt_queue and prompt_tracker.is_completed(prompt_queue[0]):
                prompt_queue.pop(0)
            if prompt_queue:
                current_prompt_idx = prompt_queue.pop(0)
                prompt_input = prompt_tracker.prompt_tensor(current_prompt_idx, device)
                prompt_text = prompt_tracker.prompts[current_prompt_idx][0]
                prompt_needs_boundary_flag = (
                    boundary_blocklist is not None and prompt_needs_boundary(prompt_text)
                )
                state_val = prompt_tracker.state(current_prompt_idx)
                if state_val == 1:
                    use_argmax_completion = True
                else:
                    use_argmax_completion = random.random() < 0.5
                sampling_strategy = "argmax" if use_argmax_completion else "sample"
            else:
                use_argmax_completion = random.random() < 0.5
                sampling_strategy = "argmax" if use_argmax_completion else "sample"
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

        if prompt_tracker is not None and current_prompt_idx is not None:
            matched, prev_state, new_state = prompt_tracker.mark_if_satisfied(
                current_prompt_idx,
                completion_ids,
                used_argmax=use_argmax_completion,
            )
            if matched and new_state > prev_state:
                expected = prompt_tracker.expected_text(current_prompt_idx)
                prompt_text = prompt_tracker.prompts[current_prompt_idx][0]
                if new_state == 2:
                    mode = "argmax"
                else:
                    mode = "sample"
                print(
                    color_text(
                        f"Prompt #{current_prompt_idx + 1} satisfied ({mode}): {prompt_text} (expected '{expected}')",
                        Colors.YELLOW,
                        bold=True,
                    )
                )

        prompt_text = tokenizer.decode_pretty(settings, torch.tensor(prompt_ids))
        completion_text = tokenizer.decode_pretty(settings, torch.tensor(completion_ids), alt=True)
        sample_prefix = color_text(prompt_text, Colors.CYAN)
        sample_suffix = color_text(completion_text, Colors.YELLOW)
        sample_render = (Colors.YELLOW if sampling_strategy == 'argmax' else Colors.CYAN) + \
                        f"{sampling_strategy}:{Colors.RESET} " + sample_prefix + sample_suffix

        def format_metric(split: str, key: str) -> str:
            value = split_metrics[split].get(key)
            if value is None:
                return "-"
            sep = ": " if key in ROW_METRIC_LOG_GROUP else ""
            return f"{sep}{value:.2f}"

        detail_keys = ROW_METRIC_LOG_KEYS

        def format_train_line() -> str:
            base = format_metric("train", "target")
            if not show_train_loss_details:
                return base
            diag = " ".join(format_metric("train", key) for key in detail_keys)
            return f"{base} {diag}"

        def format_test_line() -> str:
            base = format_metric("test", "target")
            if not show_test_loss_details:
                return base
            diag = " ".join(format_metric("test", key) for key in detail_keys)
            return f"{base} {diag}"

        train_values = format_train_line()
        test_values = format_test_line()
        if prompt_tracker is not None:
            random_only_count, solved_prompts, total_prompts = prompt_tracker.counts()
        else:
            total_prompts = len(default_prompt_entries())
            random_only_count = 0
            solved_prompts = 0
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
            "train_loss": float(split_metrics["train"].get("target", 0.0) or 0.0),
            "test_loss": float(split_metrics["test"].get("target", 0.0) or 0.0),
            "train_wall_seconds": float(total_wall_seconds),
            "unix_time": float(eval_now),
            "train_cursor": int(dataset.positions.get("train", 0)),
            "test_cursor": int(dataset.positions.get("test", 0)),
        }
        metric_keys = ["target"] + ROW_METRIC_HIST_KEYS
        for key in metric_keys:
            train_val = split_metrics["train"].get(key)
            test_val = split_metrics["test"].get(key)
            if train_val is not None:
                record[f"train_loss_{key}"] = float(train_val)
            if test_val is not None:
                record[f"test_loss_{key}"] = float(test_val)
        history_updates.append(record)

    loop_wall_total = time.time() - loop_wall_start
    loop_cpu_total = time.process_time() - loop_cpu_start
    return total_steps, history_updates, loop_wall_total, loop_cpu_total, eval_wall_total, eval_cpu_total


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
    settings: Settings,
    dataset: TextDataset,
    tokenizer: GPT2TokenizerWrapper,
    model: GRCEGPT,
    block_length: int,
    start_pos: int,
) -> None:
    tokens = dataset.looped_slice("test", start_pos, block_length)
    pretty_text = tokenizer.decode_pretty(settings, tokens)
    print(color_text(f"Test slice @ {start_pos}:", Colors.CYAN))
    print(pretty_text)


import signal
import traceback

def grce_main(args: argparse.Namespace) -> int:
    """Dispatch the CLI command selected by :func:`grce_cli_args`.

    Handles corpus management, training/reporting flow, and subcommands such
    as ``size``. When running training it constructs the model/tokenizer and
    calls :func:`train_model`.
    """
    settings = Settings(args)
    context_dropout_interval = 1

    def parse_layer_list(value: str, flag: str) -> list[int]:
        if not value:
            return []
        try:
            entries = [int(part) for part in value.split(",") if part]
        except ValueError as exc:
            raise ValueError(f"{flag} must be a comma-separated list of integers") from exc
        return entries

    args.drop_layers = parse_layer_list(args.drop_layers, "--drop-layers")
    args.add_layers = parse_layer_list(args.add_layers, "--add-layers")
    if (args.drop_layers or args.add_layers) and not args.import_model:
        raise ValueError("--drop-layers/--add-layers are only valid with --import-model")
    if args.trim_model and not args.import_model:
        raise ValueError("--trim-model is only valid with --import-model")
    args.prompt = normalize_prompt(args.prompt)
    torch.manual_seed(42)
    random.seed(42)

    class TimeoutAlarm(Exception):
        pass

    timeout_seconds = max(0.0, float(getattr(args, "timeout", 0.0)))
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
            raise TimeoutAlarm()

        prev_sigalrm_handler = signal.signal(signal.SIGALRM, handle_timeout)
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
            timeout_method = "setitimer"
        except AttributeError:
            timeout_method = "alarm"
            signal.alarm(max(1, int(math.ceil(timeout_seconds))))

    selected_action = args.command

    checkpoint_override_payload: dict | None = None
    checkpoint_override_config: ModelConfig | None = None
    checkpoint_override_tokenizer_json: str | None = None
    if args.pt:
        if selected_action == "train":
            raise ValueError("--pt is only supported for inference/debug commands")
        if args.import_model:
            raise ValueError("--pt cannot be combined with --import-model")
        skip_checkpoint_load = (
            (selected_action == "corpus" and getattr(args, "corpus_init", False))
            or selected_action == "create"
        )
        if not skip_checkpoint_load:
            if not args.pt.exists():
                raise FileNotFoundError(f"Checkpoint {args.pt} not found")
            checkpoint_override_payload = torch.load(
                args.pt, map_location="cpu", weights_only=False
            )
            saved_config = checkpoint_override_payload.get("config")
            if saved_config is None:
                raise ValueError(
                    "Checkpoint lacks config metadata; re-save it with the latest format."
                )
            saved_config = dict(saved_config)
            legacy_xctx = bool(saved_config.pop("grce_xctx", False))
            if "n_xctx" not in saved_config:
                if legacy_xctx:
                    saved_config["n_xctx"] = int(saved_config.get("n_grce", 0))
                    saved_config["n_grce"] = 0
                else:
                    saved_config["n_xctx"] = 0
            checkpoint_override_config = ModelConfig(**saved_config)
            checkpoint_override_tokenizer_json = checkpoint_override_payload.get(
                "tokenizer_json"
            )
            args.block_size = checkpoint_override_config.block_size
            if not getattr(args, "_block_length_defined", False):
                args.block_length = args.block_size
            elif args.block_length > args.block_size:
                raise ValueError("--block-length cannot exceed checkpoint block size")
            args.n_layer = checkpoint_override_config.n_layer
            args.n_head = checkpoint_override_config.n_head
            args.n_embd = checkpoint_override_config.n_embd
            args.n_grce = checkpoint_override_config.n_grce
            args.n_xctx = checkpoint_override_config.n_xctx
            args.dropout = checkpoint_override_config.dropout
            args.detach_span = checkpoint_override_config.detach_span
            args.no_detach_ctx = not checkpoint_override_config.detach_context
            args.detach_layer = checkpoint_override_config.detach_layer
            args.vocab_size = checkpoint_override_config.vocab_size

    ansi_file = None
    try:
        orig_stdout, orig_stderr, log_file = sys.stdout, sys.stderr, None

        data_dir = pathlib.Path(args.data)
        train_path = data_dir / f"{args.corpus}-train.txt.gz"
        test_path = data_dir / f"{args.corpus}-test.txt.gz"
        model_dir = pathlib.Path(args.model)
        if not model_dir.exists():
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

        must_build_tokenizer = (
            selected_action == "corpus" and getattr(args, "corpus_init", False)
        )
        train_cache_path = data_dir / f"{args.corpus}_tokens_train_{args.vocab_size}.pt"
        test_cache_path = data_dir / f"{args.corpus}_tokens_test_{args.vocab_size}.pt"
        need_corpus_for_create = False
        if selected_action == "create":
            need_corpus_for_create = (
                not train_cache_path.exists() or not test_cache_path.exists()
            )
        must_build_tokenizer = must_build_tokenizer or need_corpus_for_create

        if must_build_tokenizer:
            full_train_text = load_text_file(train_path)
            full_test_text = load_text_file(test_path)
        else:
            full_train_text = None
            full_test_text = None
            if not train_cache_path.exists():
                raise FileNotFoundError(
                    f"Train token cache {train_cache_path} not found; run 'corpus --init' first."
                )
            if not test_cache_path.exists():
                raise FileNotFoundError(
                    f"Test token cache {test_cache_path} not found; run 'corpus --init' first."
                )

        tokenizer_key = f"{args.corpus}_vocab_{args.vocab_size}"
        tokenizer_path = data_dir / f"{tokenizer_key}.json"
        if selected_action == "create" and not tokenizer_path.exists():
            must_build_tokenizer = True
        tokenizer_json = checkpoint_override_tokenizer_json
        if tokenizer_json is None:
            if tokenizer_path.exists():
                tokenizer_json = tokenizer_path.read_text(encoding="utf-8")
            elif not must_build_tokenizer:
                raise FileNotFoundError(
                    f"Tokenizer cache {tokenizer_path} not found; run 'corpus --init' first or supply --pt."
                )
        print(color_text(f"Tokenizer: {tokenizer_path}", Colors.BLUE))
        tok_timer = Timer().start()
        vocab_source = full_train_text or ""
        reserved_tokens = 1 + len(GPT2TokenizerWrapper.EXTRA_SPECIAL_TOKENS)
        if args.vocab_size <= reserved_tokens:
            raise ValueError(
                f"--vocab-size must exceed reserved tokens ({reserved_tokens}); got {args.vocab_size}"
            )
        target_vocab = max(0, args.vocab_size - reserved_tokens)
        tokenizer = GPT2TokenizerWrapper(
            vocab_source,
            tokenizer_path,
            target_vocab,
            pretrained_json=tokenizer_json,
        )
        expected_vocab_size = args.vocab_size
        actual_vocab_size = tokenizer.vocab_size
        if actual_vocab_size != expected_vocab_size:
            raise ValueError(
                "Tokenizer size mismatch: expected "
                f"{expected_vocab_size} tokens but found {actual_vocab_size}. "
                "Delete the cached tokenizer and re-run 'corpus --init'."
            )
        tokenizer_json = tokenizer_json or tokenizer_path.read_text(encoding="utf-8")

        newline_token_id = None
        newline_tokens = tokenizer.tokenizer.encode("\n", add_special_tokens=False)
        if newline_tokens:
            newline_token_id = newline_tokens[0]

        enforce_boundary_guard = not args.no_boundary
        boundary_blocklist = (
            tokenizer.leading_alpha_token_ids if enforce_boundary_guard else None
        )
        default_prompt_boundary = (
            enforce_boundary_guard
            and boundary_blocklist is not None
            and prompt_needs_boundary(args.prompt)
        )

        train_tokens, train_text, train_bytes, train_inserts = load_or_prepare_tokens(
            "train",
            train_path,
            full_train_text,
            train_cache_path,
            tokenizer,
            seed=1234,
        )

        test_tokens, test_text, test_bytes, test_inserts = load_or_prepare_tokens(
            "test",
            test_path,
            full_test_text,
            test_cache_path,
            tokenizer,
            seed=5678,
        )

        if train_text is None and train_bytes == 0:
            train_bytes = len(train_tokens)  # fallback when text absent
        if test_text is None and test_bytes == 0:
            test_bytes = len(test_tokens)

        train_token_count = int(train_tokens.numel())
        test_token_count = int(test_tokens.numel())
        print(
            color_text(
                f"Corpus size: {train_token_count:,} train tokens, {test_token_count:,} test tokens",
                Colors.CYAN,
            )
        )
        tok_summary = (
            color_text(
                f"[tokenizer time (wall/cpu/gpu/vram)]",
                Colors.CYAN,
            ) + f" {tok_timer.stop()}\n"
        )
        print(tok_summary)

        dataset = TextDataset(
            train_tokens=train_tokens,
            test_tokens=test_tokens,
            train_text=train_text,
            test_text=test_text,
            train_bytes=train_bytes,
            test_bytes=test_bytes,
            train_path=train_path,
            test_path=test_path,
        )

        def count_eval_calls(steps: int, interval: int) -> int:
            if steps <= 0:
                return 0
            eval_steps = {1, steps}
            if interval > 0:
                current = interval
                while current <= steps:
                    eval_steps.add(current)
                    current += interval
            return len(eval_steps)


        def emit_range(label: str, tokens: torch.Tensor, spec: str) -> None:
            start, end = parse_range_arg(spec)
            total = int(tokens.numel())
            if total == 0:
                print(color_text(f"{label} corpus is empty", Colors.MAGENTA))
                return
            if start < 0 or end < 0 or start >= total or end >= total:
                raise ValueError(f"{label} range {start}-{end} is outside 0-{total - 1}")
            subset = tokens[start : end + 1].tolist()
            print(
                color_text(
                    f"{label} tokens {start}-{end} (count {len(subset)}):",
                    Colors.CYAN,
                )
            )
            chunk_size = 128
            for offset in range(0, len(subset), chunk_size):
                chunk_tokens = subset[offset : offset + chunk_size]
                text = tokenizer.decode(torch.tensor(chunk_tokens))
                print(text)

        if selected_action == "corpus":
            actions_done = False
            if getattr(args, "corpus_init", False):
                print(
                    color_text(
                        "Tokenizer initialized and token caches updated; run 'train' to build a model.",
                        Colors.GREEN,
                    )
                )
                actions_done = True
            if getattr(args, "corpus_print_train", None):
                emit_range("Train", train_tokens, args.corpus_print_train)
                actions_done = True
            if getattr(args, "corpus_print_test", None):
                emit_range("Test", test_tokens, args.corpus_print_test)
                actions_done = True
            if not actions_done:
                print(color_text("No corpus action selected", Colors.YELLOW))
            return

        if args.n_xctx > 0 and args.n_xctx % max(1, args.n_layer) != 0:
            raise ValueError("--n-xctx must be divisible by --n-layer")
        config = checkpoint_override_config or ModelConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            n_grce=args.n_grce,
            n_xctx=args.n_xctx,
            dropout=args.dropout,
            detach_span=max(0, args.detach_span),
            detach_context=(not args.no_detach_ctx),
            detach_layer=max(-1, args.detach_layer),
        )
        model_tag = build_model_tag(config)
        for extra_tag in args.tag:
            cleaned = re.sub(r"[^0-9A-Za-z]+", "", extra_tag)
            if cleaned:
                model_tag += f"_{cleaned}"
        if args.pt:
            model_path = args.pt
            log_path = args.pt.with_suffix(".log")
            print(color_text(f"Model (--pt): {model_path}", Colors.CYAN))
            print(color_text(f"Logfile (--pt): {log_path}", Colors.BLUE))
        else:
            prefix = f"{args.corpus}_model_"
            model_path = model_dir / f"{prefix}{model_tag}.pt"
            log_path = model_dir / f"{prefix}{model_tag}.log"
        print(color_text(f"Model: {model_path}", Colors.CYAN))
        print(color_text(f"Logfile: {log_path}", Colors.BLUE))
        requires_checkpoint = selected_action in {"train", "report", "test"}
        if selected_action == "create" and model_path.exists():
            raise FileExistsError(
                f"Checkpoint {model_path} already exists; delete it or pick a new --model directory."
            )
        if requires_checkpoint and not model_path.exists():
            raise FileNotFoundError(
                f"Checkpoint {model_path} not found; run 'create' first to initialize it."
            )

        if selected_action == "prompts":
            target_path = Path(args.target) if args.target else model_path
            if not target_path.exists():
                raise FileNotFoundError(f"Checkpoint {target_path} not found")
            payload = torch.load(target_path, map_location="cpu", weights_only=False)
            prompt_state = payload.get("prompt_state") or empty_prompt_state()
            tracker = PromptTracker(tokenizer, prompt_state)
            entries = list(tracker.prompts)
            statuses = list(tracker.status)

            def normalize_statuses() -> None:
                nonlocal statuses
                length = len(entries)
                if len(statuses) < length:
                    statuses.extend([0] * (length - len(statuses)))
                elif len(statuses) > length:
                    statuses = statuses[:length]

            normalize_statuses()
            changed = False
            if getattr(args, "reset", False):
                entries = default_prompt_entries()
                statuses = [0] * len(entries)
                changed = True
            if getattr(args, "clear", False):
                entries = []
                statuses = []
                changed = True
            removes = sorted(set(getattr(args, "remove", [])), reverse=True)
            for idx in removes:
                if idx is None:
                    continue
                try:
                    intval = int(idx)
                except (TypeError, ValueError):
                    continue
                if 0 <= intval < len(entries):
                    entries.pop(intval)
                    if len(statuses) > intval:
                        statuses.pop(intval)
                    changed = True
                else:
                    print(
                        color_text(
                            f"Prompt index {intval} out of range; ignoring remove request.",
                            Colors.YELLOW,
                        )
                    )
            add_pair = getattr(args, "add", None)
            if add_pair is not None:
                prompt_text, expected_text = add_pair
                entries.append((prompt_text, expected_text))
                statuses.append(0)
                changed = True
            normalize_statuses()
            if changed:
                new_state = build_prompt_state(entries, statuses)
                payload["prompt_state"] = new_state
                torch.save(payload, target_path)
                print(
                    color_text(
                        f"Updated prompts in {target_path}",
                        Colors.GREEN,
                    )
                )
            show_list = args.list or (
                not getattr(args, "reset", False)
                and not getattr(args, "clear", False)
                and not removes
                and add_pair is None
            )
            if show_list:
                if not entries:
                    print(color_text("No prompts stored in checkpoint", Colors.MAGENTA))
                else:
                    print(color_text(f"Prompts in {target_path}:", Colors.CYAN))
                    for idx, (prompt_text, expected_text) in enumerate(entries):
                        status_val = statuses[idx] if idx < len(statuses) else 0
                        if status_val >= 2:
                            status_label = color_text("argmax", Colors.GREEN)
                        elif status_val == 1:
                            status_label = color_text("sample", Colors.YELLOW)
                        else:
                            status_label = color_text("pending", Colors.RED)
                        print(
                            color_text(f"#{idx}: ", Colors.CYAN)
                            + status_label
                            + color_text(
                                f" prompt='{prompt_text}' expected='{expected_text}'",
                                Colors.CYAN,
                            )
                        )
            return
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
        if not args.no_ansi:
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

        device = torch.device(args.device)
        try:
            prompt_tokens = tokenizer.encode(args.prompt)
        except KeyError as exc:  # pragma: no cover - user misconfiguration
            raise ValueError(
                "Prompt contains characters outside the tokenizer vocabulary. "
                "Choose a simpler prompt or extend the dataset."
            ) from exc
        try:
            prompt_tokens = prompt_tokens.unsqueeze(0).to(device)
        except (AssertionError, RuntimeError) as exc:
            message = str(exc)
            if "Torch not compiled with CUDA" in message and args.device != "cpu":
                if not args.tiny:
                    raise RuntimeError(
                        "CUDA requested but not available; rerun with --device cpu or --tiny."
                    ) from exc
                print(color_text("Torch not compiled with CUDA enabled; switching to CPU", Colors.RED, bold=True))
                device = torch.device("cpu")
                args.device = "cpu"
                prompt_tokens = prompt_tokens.unsqueeze(0).to(device)
            else:
                raise
        prompt_tracker = PromptTracker(tokenizer)

        try:
            model = GRCEGPT(config).to(device)
        except (AssertionError, RuntimeError) as exc:
            message = str(exc)
            if "Torch not compiled with CUDA" in message and args.device != "cpu":
                if not args.tiny:
                    raise RuntimeError(
                        "CUDA requested but not available; rerun with --device cpu or --tiny."
                    ) from exc
                print(color_text("Torch not compiled with CUDA enabled; switching to CPU", Colors.RED, bold=True))
                device = torch.device("cpu")
                model = GRCEGPT(config).to(device)
            else:
                raise
        total_steps = 0
        loss_history: List[Dict[str, float]] = []
        total_train_wall = 0.0
        payload = checkpoint_override_payload
        if payload is None and model_path.exists():
            if args.import_model:
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
                    if "dataset" in payload:
                        dataset.load_state(payload["dataset"])
                    total_steps = int(payload.get("total_steps", 0))
                    loss_history = list(payload.get("loss_history", []))
                    total_train_wall = float(payload.get("train_wall_seconds", 0.0))
                    prompt_tracker.load_state(payload.get("prompt_state"))
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
                prompt_total = len(prompt_tracker.prompts)
                if prompt_tracker.remaining() < prompt_total:
                    solved_argmax = [
                        idx for idx, state in enumerate(prompt_tracker.status) if state == 2
                    ]
                    solved_random_only = [
                        idx for idx, state in enumerate(prompt_tracker.status) if state == 1
                    ]
                    if solved_random_only:
                        lines = []
                        for idx in solved_random_only:
                            if 0 <= idx < prompt_total:
                                text, expected = prompt_tracker.prompts[idx]
                            else:
                                continue
                            lines.append(
                                color_text(
                                    f"#{idx + 1}: '{text}' -> '{expected}'",
                                    Colors.YELLOW,
                                )
                            )
                        print(
                            "\n"
                            + color_text(
                                f"Random-only prompts ({len(solved_random_only)}/{prompt_total}):",
                                Colors.YELLOW,
                                bold=True,
                            )
                            + "\n"
                            + "\n".join(lines)
                        )
                    if solved_argmax:
                        lines = []
                        for idx in solved_argmax:
                            if 0 <= idx < prompt_total:
                                text, expected = prompt_tracker.prompts[idx]
                            else:
                                continue
                            lines.append(
                                color_text(
                                    f"#{idx + 1}: '{text}' -> '{expected}'",
                                    Colors.GREEN,
                                )
                            )
                        print(
                            "\n"
                            + color_text(
                                f"Argmax-satisfied prompts ({len(solved_argmax)}/{prompt_total}):",
                                Colors.GREEN,
                                bold=True,
                            )
                            + "\n"
                            + "\n".join(lines)
                        )
            except RuntimeError as err:
                print(color_text("Checkpoint load failed (shape mismatch); starting fresh.", Colors.RED, bold=True))
                print(color_text(str(err), Colors.RED))
        elif args.import_model:
            import_timer = Timer().start()
            if not args.import_model.exists():
                raise FileNotFoundError(f"Import checkpoint {args.import_model} not found")
            source_state, meta = load_checkpoint_payload(args.import_model, device)
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
            print(color_text(f"Importing weights from {args.import_model}", Colors.GREEN))
            mapping = build_layer_mapping(
                src_layers,
                config.n_layer,
                args.drop_layers,
                args.add_layers,
                allow_trim=args.trim_model,
            )
            apply_imported_state(
                model,
                source_state,
                allow_trim=args.trim_model,
                mapping=mapping,
            )
            total_steps = int(meta.get("total_steps", 0))
            total_train_wall = float(meta.get("train_wall_seconds", 0.0))
            loss_history = []
            write_timer = Timer().start()
            torch.save(
                {
                    "model": model.state_dict(),
                    "dataset": dataset.state_dict(),
                    "total_steps": total_steps,
                    "loss_history": loss_history,
                    "config": asdict(config),
                    "train_wall_seconds": total_train_wall,
                    "prompt_state": prompt_tracker.serialize() if prompt_tracker else None,
                    "tokenizer_json": tokenizer_json,
                },
                model_path,
            )
            print(
                color_text(
                    f"[import] total steps: {total_steps}; time spent (wall/cpu/gpu/vram): {import_timer.stop()}; writing model: {write_timer.stop()}",
                    Colors.CYAN,
                )
            )
            return
        if selected_action == "create":
            checkpoint_payload = {
                "model": model.state_dict(),
                "dataset": dataset.state_dict(),
                "total_steps": 0,
                "loss_history": [],
                "config": asdict(config),
                "train_wall_seconds": 0.0,
                "prompt_state": prompt_tracker.serialize(),
                "tokenizer_json": tokenizer_json,
            }
            torch.save(checkpoint_payload, model_path)
            print(
                color_text(
                    f"Created new checkpoint at {model_path}; run 'train' to begin training.",
                    Colors.GREEN,
                )
            )
            return

        if selected_action == "report":
            run_report_mode(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                sample_len=args.generate,
                count=args.report_count,
                device=device,
                suppress_newlines=args.no_newlines,
                newline_token_id=newline_token_id,
                default_prompt_boundary=default_prompt_boundary,
                boundary_blocklist=boundary_blocklist,
            )
            return

        if selected_action == "test":
            run_test_slice(
                settings=settings,
                dataset=dataset,
                tokenizer=tokenizer,
                model=model,
                block_length=args.block_length,
                start_pos=args.test_start,
            )
            return

        acc_train_wall = 0.0
        acc_train_cpu = 0.0
        acc_eval_wall = 0.0
        acc_eval_cpu = 0.0
        for cycle in range(1, args.cycles + 1):
            cycle_wall = time.time()
            tags = ["GPT"]
            plus_tags: list[str] = []
            minus_tags: list[str] = []
            if args.n_grce > 0:
                plus_tags.append("+GRCE")
            else:
                minus_tags.append(" wo/GRCE")
            if args.n_xctx > 0:
                plus_tags.append("+XCTX")
            else:
                minus_tags.append(" wo/XCTX")
            label = "".join(tags + plus_tags + minus_tags)
            hours = total_train_wall / 3600.0
            days = hours / 24.0
            train_chars_cycle = (args.block_length + 1) * args.batch_size * args.steps
            eval_calls = max(1, count_eval_calls(args.steps, args.eval_interval))
            test_chars_cycle = (
                (args.block_length + 1)
                * args.batch_size
                * eval_calls
            )
            train_start = int(dataset.positions.get("train", 0))
            test_start = int(dataset.positions.get("test", 0))
            dataset.prepare_cycle("train", train_chars_cycle)
            dataset.prepare_cycle("test", test_chars_cycle)

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
            print(
                color_text(
                    f"Corpus slices: train tokens {train_range}, test tokens {test_range}",
                    Colors.CYAN,
                )
            )
            print(color_text(f"Model: {model_path}", Colors.CYAN))
            print(
                color_text(
                    f"[{label}] Training Cycle {cycle}/{args.cycles}. "
                    f"Total training so far: {total_steps} steps, {hours:.2f} hours ({days:.2f} days)",
                    Colors.BLUE,
                )
            )

            (
                total_steps,
                updates,
                cycle_wall_elapsed,
                cycle_cpu_elapsed,
                eval_wall_total,
                eval_cpu_total,
            ) = train_model(
                settings,
                model,
                dataset,
                device,
                args.steps,
                args.block_length,
                args.batch_size,
                args.eval_interval,
                total_steps,
                prompt_tokens,
                args.generate,
                tokenizer,
                suppress_newlines=args.no_newlines,
                newline_token_id=newline_token_id,
                prompt_tracker=prompt_tracker,
                reset_prompt_queue=args.reset_prompt_each_cycle,
                cycle_wall_start=cycle_wall,
                base_wall_seconds=total_train_wall,
                show_time=args.time,
                default_prompt_boundary=default_prompt_boundary,
                boundary_blocklist=boundary_blocklist,
                show_train_loss_details=args.train_loss_details,
                show_test_loss_details=not args.no_test_loss_details,
            )
            loss_history.extend(updates)

            train_wall = cycle_wall_elapsed
            train_cpu = cycle_cpu_elapsed
            eval_wall = eval_wall_total
            eval_cpu = eval_cpu_total
            pure_train_wall = max(0.0, train_wall - eval_wall)
            pure_train_cpu = max(0.0, train_cpu - eval_cpu)
            acc_train_wall += pure_train_wall
            acc_train_cpu += pure_train_cpu
            acc_eval_wall += eval_wall
            acc_eval_cpu += eval_cpu
            total_train_wall += train_wall
            torch.save(
                {
                    "model": model.state_dict(),
                    "dataset": dataset.state_dict(),
                    "total_steps": total_steps,
                    "loss_history": loss_history,
                    "config": asdict(config),
                    "train_wall_seconds": total_train_wall,
                    "prompt_state": prompt_tracker.serialize() if prompt_tracker else None,
                    "tokenizer_json": tokenizer_json,
                },
                model_path,
            )
            cycle_part = color_text(f"[cycle {cycle}]", Colors.CYAN)
            train_part = color_text(
                f" train: wall={pure_train_wall:.2f}s cpu={pure_train_cpu:.2f}s;",
                Colors.MAGENTA,
            )
            eval_part = color_text(
                f" eval: wall={eval_wall:.2f}s cpu={eval_cpu:.2f}s;",
                Colors.GREEN,
            )
            updated_part = color_text(" model updated.", Colors.YELLOW)
            print(cycle_part + train_part + eval_part + updated_part)
            def ratio_text(num: float, denom: float) -> str:
                if denom <= 0:
                    if num <= 0:
                        return "0.0"
                    return "inf"
                return f"{num / denom:.2f}"

            wall_ratio = ratio_text(acc_train_wall, acc_eval_wall)
            cpu_ratio = ratio_text(acc_train_cpu, acc_eval_cpu)
            cumulative_part = color_text("[cumulative]", Colors.CYAN)
            cum_train_part = color_text(
                f" train: wall={acc_train_wall:.2f}s cpu={acc_train_cpu:.2f}s;",
                Colors.MAGENTA,
            )
            cum_eval_part = color_text(
                f" eval: wall={acc_eval_wall:.2f}s cpu={acc_eval_cpu:.2f}s;",
                Colors.GREEN,
            )
            ratio_text = color_text(
                f" train/eval: wall={wall_ratio} cpu={cpu_ratio}",
                Colors.CYAN,
            )
            print(cumulative_part + cum_train_part + cum_eval_part + ratio_text)
            if log_file is not None:
                log_file.flush()
            if ansi_file is not None:
                ansi_file.flush()

    except KeyboardInterrupt:
        if args.debug_interrupt:
            raise
        # traceback.print_exc()
        print(color_text("Interrupted by user; exiting cleanly.", Colors.RED, bold=True))
    except TimeoutAlarm:
        # traceback.print_exc()
        print(color_text("Timeout; exiting cleanly.", Colors.RED, bold=True))

    finally:
        if timeout_method is not None:
            cancel_timeout()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        if log_file is not None:
            log_file.close()
        if ansi_file is not None:
            ansi_file.close()

    return 0

if __name__ == "__main__":
    # second entry point for "size" subcommand, now with
    # Torch imported; used only in 'size --check' mode
    if cli_args.command == "size":
        assert getattr(cli_args, "check", False)
        sys.exit(grce_cli_size(cli_args))

    # otherwise: run the big "default" main
    sys.exit(grce_main(cli_args))
