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
class ModelGeometry:
    """Holds the GPT+GRCE+XCTX model geometry."""

    vocab_size: int = 5000  # GPT-2 base supports ~50k merges; we stay small for the PoC.
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
    corpus: str = "cccclc"
    steps: int = 100
    cycles: int = 100
    batch_size: int = 256
    batch_layout: str = "random"  # random, split, stacked, or both batch composition
    rows_encode: int | None = None
    rows_decode: int | None = None
    rows_forward: int | None = None
    rows_noattn: int | None = None
    split_length: int | None = None
    split_size: int | None = None
    split_steps: int | None = None
    cols_encode: int | None = None
    cols_decode: int | None = None
    cols_forward: int | None = None
    cols_noattn: int | None = None
    eval_interval: int = 10
    dropout: float = 0.05
    detach_span: int = 0

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
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Sequence


def parse_range_arg(value: str) -> tuple[int, int]:
    """Parse ``START-END`` strings for :meth:`Runtime.cli_corpus` emitters."""

    parts = value.replace(" ", "").split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid range '{value}'. Expected format START-END.")
    start, end = int(parts[0]), int(parts[1])
    if end < start:
        raise ValueError(f"Range end {end} is smaller than start {start}.")
    return start, end


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
        "--corpus",
        type=str,
        default=DEFAULTS.corpus,
        help="Dataset base name; expects data/<name>-train.txt.gz and ...-test.txt.gz.",
    )
    generic.add_argument(
        "--data",
        type=str,
        default="data",
        help="Directory containing <corpus>-train.txt.gz and <corpus>-test.txt.gz",
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
            "Shortcut for --vocab-size 500 --batch-size 12 --block-size 6 --n-layer 3 --n-head 2 "
            "--n-embd 8 --n-grce 4 --n-xctx 9 --steps 2 --cycles 1 --eval-interval 1 --corpus simplestwiki"
        ),
    )
    model_group.add_argument(
        "--arith",
        action="store_true",
        help=(
            "Shortcut for --vocab-size 500 --batch-size 700 --block-size 64 --n-layer 10 --n-head 4 "
            "--n-embd 128 --n-grce 32 --n-xctx 720 --corpus simplearith"
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
        "--batch-layout",
        choices=["random", "split", "stacked", "both"],
        default=DEFAULTS.batch_layout,
        help="Batch composition strategy: random mix, force split/stacked, or run both",
    )
    training_group.add_argument(
        "--rows-encode",
        type=int,
        default=DEFAULTS.rows_encode,
        help="Pin the encode row count for split batches (omit for random)",
    )
    training_group.add_argument(
        "--rows-decode",
        type=int,
        default=DEFAULTS.rows_decode,
        help="Pin the decode row count for split batches (omit for random)",
    )
    training_group.add_argument(
        "--rows-forward",
        type=int,
        default=DEFAULTS.rows_forward,
        help="Pin the forward row count for split batches (omit for random)",
    )
    training_group.add_argument(
        "--rows-noattn",
        type=int,
        default=DEFAULTS.rows_noattn,
        help="Pin the no-attention row count for split batches (omit for random)",
    )
    training_group.add_argument(
        "--split-length",
        type=int,
        default=DEFAULTS.split_length,
        help="Override --block-length for split batches only",
    )
    training_group.add_argument(
        "--split-size",
        type=int,
        default=DEFAULTS.split_size,
        help="Override --batch-size for split batches only",
    )
    training_group.add_argument(
        "--split-steps",
        type=int,
        default=DEFAULTS.split_steps,
        help="Number of steps per cycle assigned to split mode (default=steps/2)",
    )
    training_group.add_argument(
        "--cols-encode",
        type=int,
        default=DEFAULTS.cols_encode,
        help="Pin the encode column count for stacked batches (omit for random)",
    )
    training_group.add_argument(
        "--cols-decode",
        type=int,
        default=DEFAULTS.cols_decode,
        help="Pin the decode column count for stacked batches (omit for random)",
    )
    training_group.add_argument(
        "--cols-forward",
        type=int,
        default=DEFAULTS.cols_forward,
        help="Pin the forward column count for stacked batches (omit for random)",
    )
    training_group.add_argument(
        "--cols-noattn",
        type=int,
        default=DEFAULTS.cols_noattn,
        help="Pin the no-attention column count for stacked batches (omit for random)",
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
    # Subcommand args parser for "corpus"

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


    # --------------------------------------------------------
    # Run the args parser

    args = parser.parse_args()

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

    if args.arith:
        if not flag_present("--vocab-size"):
            args.vocab_size = 313
        if not flag_present("--batch-size"):
            args.batch_size = 128
        if not flag_present("--block-size"):
            args.block_size = 64
        if not flag_present("--n-layer"):
            args.n_layer = 10
        if not flag_present("--n-head"):
            args.n_head = 4
        if not flag_present("--n-embd"):
            args.n_embd = 128
        if not flag_present("--n-grce"):
            args.n_grce = 32
        if not flag_present("--n-xctx"):
            args.n_xctx = 720
        if not flag_present("--corpus"):
            args.corpus = "simplearith"

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


    # --------------------------------------------------------
    # Parse and check "corpus" sub-command args

    if args.command == "corpus":
        has_corpus_action = bool(
            args.corpus_init
            or args.corpus_init_tokenizer
            or args.corpus_print_train is not None
            or args.corpus_print_test is not None
        )
        if not has_corpus_action:
            parser.error("corpus command requires --init-tokenizer, --init and/or --print-* options")


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

    for attr in ("rows_encode", "rows_decode", "rows_forward", "rows_noattn"):
        value = getattr(args, attr, None)
        if value is not None and value < 0:
            parser.error(f"--{attr.replace('_', '-')} must be non-negative")
    for attr in ("cols_encode", "cols_decode", "cols_forward", "cols_noattn"):
        value = getattr(args, attr, None)
        if value is not None and value < 0:
            parser.error(f"--{attr.replace('_', '-')} must be non-negative")
    col_pins = [args.cols_encode, args.cols_decode, args.cols_forward, args.cols_noattn]
    if all(value is not None for value in col_pins):
        total_cols = sum(int(value) for value in col_pins)
        if total_cols > args.block_size:
            parser.error("Sum of pinned --cols-* cannot exceed --block-size")
    if getattr(args, "split_length", None) is not None:
        if args.split_length <= 0:
            parser.error("--split-length must be positive")
        if args.split_length > args.block_size:
            parser.error("--split-length cannot exceed --block-size")
    if getattr(args, "split_size", None) is not None and args.split_size <= 0:
        parser.error("--split-size must be positive")
    if getattr(args, "split_steps", None) is not None:
        if args.split_steps < 0:
            parser.error("--split-steps must be non-negative")
        if args.split_steps > args.steps:
            parser.error("--split-steps cannot exceed --steps")
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


# -----------------------------------------------------------------------------
# Lightweight (non-Torch) Library Components
# -----------------------------------------------------------------------------

def load_text_file(path: pathlib.Path) -> str:
    """Load raw corpus text for tokenizer prep and CLI corpus utilities."""

    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Provide a text file path.")
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8")


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
    """Return True when :class:`PromptTracker` should append boundary tokens."""

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
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

ASCII_LETTERS = set(string.ascii_letters)
ASCII_LOWERCASE = set(string.ascii_lowercase)
TOKENIZER_TRAIN_LIMIT_BYTES = 16 * 1024 * 1024  # 16 MB of corpus text for tokenizer training

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
    """Return the built-in prompt catalog used by :class:`PromptTracker`."""

    return [(prompt, expected) for prompt, expected in PROMPT_GOALS]


def serialized_prompts(entries: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Convert prompt tuples into checkpoint-friendly dictionaries."""

    return [{"prompt": prompt, "expected": expected} for prompt, expected in entries]


def empty_prompt_state(entries: list[tuple[str, str]] | None = None) -> dict:
    """Create a fresh tracker state for :class:`Runtime` prompt persistence."""

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
    """Combine prompts and status overrides for :class:`PromptTracker`."""

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
    """Normalize raw text prior to tokenizer training in :class:`Runtime`."""

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


def _limit_training_text_bytes(
    text: str,
    limit_bytes: int = TOKENIZER_TRAIN_LIMIT_BYTES,
) -> str:
    """Clamp tokenizer training data size for :class:`GPT2TokenizerWrapper`."""

    if limit_bytes <= 0 or not text:
        return text
    try:
        raw = text.encode("utf-8")
    except UnicodeEncodeError:
        return text[:limit_bytes]
    if len(raw) <= limit_bytes:
        return text
    truncated = raw[:limit_bytes]
    return truncated.decode("utf-8", errors="ignore")


# -----------------------------------------------------------------------------
# Data Utilities
# -----------------------------------------------------------------------------


class GPT2TokenizerWrapper:
    """Tokenizer shim used wherever :class:`Runtime` needs GPT-2 style BPE."""

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
        #self.byte_fallback_encodings = self._build_byte_fallback_encodings()

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
        byte_level_alphabet = ByteLevel.alphabet()
        if byte_level_alphabet:
            initial_alphabet = byte_level_alphabet
        else:
            byte_values = sorted(set(train_text.encode("utf-8")))
            observed_alphabet = [chr(b) for b in byte_values]
            initial_alphabet = observed_alphabet or [chr(b) for b in range(256)]
        # Reserve one additional slot beyond the requested vocab size to compensate
        # for the underlying trainer implicitly injecting an end-of-input token.
        trainer_vocab_size = vocab_size + 1
        trainer = BpeTrainer(
            vocab_size=trainer_vocab_size,
            min_frequency=2,
            special_tokens=[],
            initial_alphabet=initial_alphabet,
        )
        limited_text = _limit_training_text_bytes(train_text)
        sanitized = _restrict_bpe_training_text(limited_text)
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

    def _build_byte_fallback_encodings(self) -> list[tuple[int, ...]]:
        encodings: list[tuple[int, ...]] = []
        for value in range(256):
            byte_text = bytes([value]).decode("latin-1")
            try:
                tokens = self.tokenizer.encode(byte_text, add_special_tokens=False)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ValueError(
                    f"Tokenizer could not encode byte {value}; delete {self.cache_path} and rebuild the tokenizer cache"
                ) from exc
            cleaned = tuple(int(tok) for tok in tokens)
            if not cleaned:
                raise ValueError(
                    f"Tokenizer produced no tokens for byte {value}; delete {self.cache_path} and rebuild the tokenizer cache"
                )
            if any(tok in self.special_ids for tok in cleaned):
                raise ValueError(
                    f"Tokenizer fallback for byte {value} relies on a special token; delete {self.cache_path} and rebuild the tokenizer cache"
                )
            encodings.append(cleaned)
        return encodings

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

    def decode_pretty(self, args: Args, tokens: torch.Tensor, color: str = Colors.MAGENTA, altcolor: str = Colors.GREEN, alt: bool = False) -> str:
        if alt: color, altcolor =  Colors.YELLOW, Colors.CYAN
        parts = []
        for tok in tokens.tolist():
            s = self.decode_one(tok)
            assert s, "got empty token"
            if s == " " or " " in s[1:]: s = s.replace(" ", FANCY_SPACE)
            s = s.replace("\n", FANCY_ENTER if args.escape_newline_tokens else FANCY_ENTER.replace(" ", "\n"))
            parts.append(color + s + Colors.RESET)
            color, altcolor = altcolor, color
        return "".join(parts)


class PromptTracker:
    """Maintains evaluation prompts reused by :class:`Runtime` diagnostics."""

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
    """Holds rolling corpus state for :class:`Runtime` training/eval loops."""

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





def load_or_prepare_tokens(
    split: str,
    text_path: str,
    text: str | None,
    cache_path: pathlib.Path,
    tokenizer: GPT2TokenizerWrapper,
    seed: int,
) -> Tuple[torch.Tensor, str | None, int, int]:
    """Load cached token tensors or create them for :class:`Runtime` setups."""

    if cache_path.exists():
        payload = torch.load(cache_path)
        tokens = payload["tokens"].long()
        bytes_count = int(payload.get("bytes", 0))
        print(color_text(f"Loaded cached {split} tokens from {cache_path}", Colors.YELLOW))
        return tokens, text, bytes_count

    print(color_text(f"Tokenizing raw {split} data: {text_path}...", Colors.BLUE))

    if text is None:
        raise FileNotFoundError(
            f"No cached tokens at {cache_path} and source text missing for {split}."
        )
    tokens = tokenizer.encode_corpus(text).type(torch.uint16)
    bytes_count = len(text.encode("utf-8"))
    torch.save({"tokens": tokens, "bytes": bytes_count}, cache_path)
    print(color_text(f"Saved {split} token cache to {cache_path}", Colors.YELLOW))
    return tokens, text, bytes_count


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

    def forward(
        self,
        x: torch.Tensor,
        *,
        block_bias: torch.Tensor | None = None,
        kv_cache_sources: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
        qh_query_callback=None,
        attn_mode: str = "decode",
        layer_idx: int = 0,
        record_mask: bool = False,
        attention_disabled_rows: torch.Tensor | None = None,
        attention_dropout_positions: torch.Tensor | None = None,
        full_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_norm = self.ln1(x)
        if block_bias is not None:
            attn_norm = attn_norm + block_bias
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
        pre_ff = self.ln2(x)
        if block_bias is not None:
            pre_ff = pre_ff + block_bias
        ff_out, mask = self.ff(pre_ff, record_mask=record_mask)
        x = x + ff_out
        return x, mask, kv_pair

    def forward_incremental(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        *,
        block_bias: torch.Tensor | None = None,
        record_mask: bool = False,
        attention_disabled_rows: torch.Tensor | None = None,
        puncture_mask: torch.Tensor | None = None,
        write_cache: bool = True,
    ) -> tuple[torch.Tensor, LayerCache, torch.Tensor | None]:
        attn_norm = self.ln1(x)
        if block_bias is not None:
            attn_norm = attn_norm + block_bias
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
        pre_ff = self.ln2(x)
        if block_bias is not None:
            pre_ff = pre_ff + block_bias
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
    merged: list[tuple[torch.Tensor, torch.Tensor]] = []
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
        bias_list_in: Sequence[torch.Tensor] | None = None,
        kv_cache_list_in: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]] | None = None,
        *,
        mode: str = "decode",
        qh_query_callback=None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        rows, cols, _ = x.shape
        bias_tensor = _merge_bias_list(
            bias_list_in or [],
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
            layer_bias = None
            if bias_tensor is not None:
                layer_bias = bias_tensor[:, :, layer_idx, :]
            kv_sources = layer_kv_sources[layer_idx] or None
            current, _, kv_pair = block(
                current,
                block_bias=layer_bias,
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
        bias_list_in: Sequence[torch.Tensor] | None = None,
        kv_cache_list_in: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]] | None = None,
        *,
        qh_query_callback=None,
        mode: str = "decode",
    ) -> tuple[torch.Tensor, list[torch.Tensor], list]:
        output, samples, kv_out = self.core.forward_grid(
            x,
            bias_list_in=bias_list_in,
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
        self.output_norm = nn.LayerNorm(self.context_dim)

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
        mixed = self.mix_norm(combined)
        mlp_out = self.mlp_down(F.gelu(self.mlp_up(mixed)))
        return self.output_norm(mixed + mlp_out)

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
        combined = xctx_state + fused
        squeezed = self.mix_norm(self.mix_down(combined))
        mlp = F.gelu(self.mix_up(squeezed))
        projected = self.mix_proj(mlp)
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
        bias_list_in: Sequence[torch.Tensor] | None = None,
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
                bias_list_in=bias_list_in,
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
            column_biases: list[torch.Tensor] = []
            if bias_list_in:
                for bias in bias_list_in:
                    if bias is None or bias.size(1) == 0:
                        continue
                    if bias.size(1) == 1:
                        column_biases.append(bias)
                    elif col < bias.size(1):
                        column_biases.append(bias[:, col : col + 1, :, :])
            if self.grce is not None and grce_state is not None:
                column_biases.append(self.grce.bias_forward(grce_state))
            if self.xctx is not None and xctx_state is not None:
                column_biases.append(self.xctx.bias_forward(xctx_state))
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
                bias_list_in=column_biases,
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
        bias_list_in: Sequence[torch.Tensor] | None,
        kv_cache_list_in: Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]] | None,
        qh_query_callback=None,
        mode: str,
        detach_samples_span: int,
        detach_grce_span: int,
        detach_xctx_span: int,
        detach_internal_kv_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]]]:
        rows, cols, _ = x.shape
        grid_biases = list(bias_list_in or [])
        if self.grce is not None and grce_state is not None:
            grid_biases.append(self.grce.bias_forward(grce_state))
        if self.xctx is not None and xctx_state is not None:
            grid_biases.append(self.xctx.bias_forward(xctx_state))
        output, samples, kv_pairs = self.core.forward_grid(
            x,
            bias_list_in=grid_biases,
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


def allocate_split_rows(
    total_rows: int,
    rows_encode: int | None,
    rows_decode: int | None,
    rows_forward: int | None,
    rows_noattn: int | None,
) -> dict[str, int]:
    """Randomly allocate row counts per mode subject to optional pins."""

    pinned = {
        "encode": rows_encode,
        "decode": rows_decode,
        "forward": rows_forward,
        "noattn": rows_noattn,
    }
    total = max(len(pinned), int(total_rows))
    pinned_total = sum(int(v) for v in pinned.values() if v is not None)
    all_pinned = all(value is not None for value in pinned.values())
    if all_pinned:
        total = pinned_total
    elif pinned_total > total:
        raise ValueError("Sum of pinned --rows-* exceeds batch size")
    remaining = total - pinned_total
    allocations = {mode: int(value) if value is not None else 0 for mode, value in pinned.items()}
    for mode in pinned:
        if pinned[mode] is None and allocations[mode] <= 0:
            need = 1
            allocations[mode] = need
            remaining -= need
    unspecified = [mode for mode, value in pinned.items() if value is None]
    for idx, mode in enumerate(unspecified):
        if idx == len(unspecified) - 1:
            take = remaining
        else:
            low = max(1, total // 8)
            high = max(low, total // 2)
            take = random.randint(low, high)
            take = min(max(1, take), remaining)
        allocations[mode] += take
        remaining -= take
    return allocations


def allocate_stacked_cols(
    total_cols: int,
    cols_encode: int | None,
    cols_decode: int | None,
    cols_forward: int | None,
    cols_noattn: int | None,
) -> dict[str, int]:
    """Randomly allocate column counts per mode subject to optional pins."""

    pinned = {
        "encode": cols_encode,
        "decode": cols_decode,
        "forward": cols_forward,
        "noattn": cols_noattn,
    }
    total = max(len(pinned), int(total_cols))
    pinned_total = sum(int(v) for v in pinned.values() if v is not None)
    all_pinned = all(value is not None for value in pinned.values())
    if all_pinned:
        total = pinned_total
    elif pinned_total > total:
        raise ValueError("Sum of pinned --cols-* exceeds block length")
    remaining = total - pinned_total
    allocations = {mode: int(value) if value is not None else 0 for mode, value in pinned.items()}
    for mode in pinned:
        if pinned[mode] is None and allocations[mode] <= 0:
            allocations[mode] = 1
            remaining -= 1
    unspecified = [mode for mode, value in pinned.items() if value is None]
    for idx, mode in enumerate(unspecified):
        if idx == len(unspecified) - 1:
            take = remaining
        else:
            low = max(1, total // 8)
            high = max(low, total // 2)
            take = random.randint(low, high)
            take = min(max(1, take), remaining)
        allocations[mode] += take
        remaining -= take
    return allocations


def build_cycle_layouts(step_count: int, mode: str, split_steps: int | None) -> list[str]:
    """Return a per-step layout list honoring explicit or random selection."""

    if mode == "split":
        return ["split"] * step_count
    if mode == "stacked":
        return ["stacked"] * step_count
    if mode == "both":
        return ["both"] * step_count
    if mode != "random":
        raise ValueError(f"Unknown batch layout: {mode}")
    if split_steps is None:
        split_count = step_count // 2
    else:
        split_count = max(0, min(step_count, int(split_steps)))
    stacked_count = step_count - split_count
    layouts = ["split"] * split_count + ["stacked"] * stacked_count
    random.shuffle(layouts)
    return layouts


def effective_split_rows(args: Args, base_rows: int) -> int:
    pinned = [args.rows_encode, args.rows_decode, args.rows_forward, args.rows_noattn]
    if args.split_size is not None:
        return max(0, int(args.split_size))
    if all(value is not None for value in pinned):
        return max(0, sum(int(value) for value in pinned))
    return max(0, int(base_rows))


def effective_split_length(args: Args, base_length: int) -> int:
    if args.split_length is not None:
        return max(1, int(args.split_length))
    return max(1, int(base_length))


def effective_stacked_length(args: Args, base_length: int) -> int:
    pinned = [args.cols_encode, args.cols_decode, args.cols_forward, args.cols_noattn]
    if all(value is not None for value in pinned):
        return max(1, sum(int(value) for value in pinned))
    return max(1, int(base_length))

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


def evaluate_single_split_batch(
    args: Args,
    model: GRCEGPT,
    dataset: TextDataset,
    split: str,
    block_length: int,
    batch_size: int,
    device: torch.device,
) -> EvalBatchStats:
    """Evaluate a split batch (rows per mode) and report per-mode losses."""

    metrics: dict[str, float | None] = {mode: None for mode in BATCH_MODES}
    metrics["target"] = None
    loss_sums = {mode: 0.0 for mode in BATCH_MODES}
    loss_sums["target"] = 0.0
    token_counts = {mode: 0 for mode in BATCH_MODES}
    token_counts["target"] = 0
    split_total_rows = effective_split_rows(args, batch_size)
    split_block_length = effective_split_length(args, block_length)
    allocations = allocate_split_rows(
        split_total_rows,
        args.rows_encode,
        args.rows_decode,
        args.rows_forward,
        args.rows_noattn,
    )
    total_loss = 0.0
    total_tokens = 0
    for mode in BATCH_MODES:
        rows = allocations.get(mode, 0)
        if rows <= 0:
            continue
        xb, yb = dataset.get_batch(split, split_block_length, rows, device)
        position_offsets = sample_position_offsets(
            rows,
            model.config.block_size,
            split_block_length,
            device,
        )
        logits, _, _ = model.forward_autoreg(
            xb,
            targets=yb,
            mode=mode,
            position_offsets=position_offsets,
        )
        loss_sum, token_count = loss_sum_and_token_count(
            logits, yb, last_only=(mode == "encode")
        )
        if token_count <= 0:
            continue
        loss_value = float(loss_sum.item())
        loss_sums[mode] = loss_value
        token_counts[mode] = token_count
        metrics[mode] = loss_value / token_count
        total_loss += loss_value
        total_tokens += token_count
    if total_tokens > 0:
        metrics["target"] = total_loss / total_tokens
    loss_sums["target"] = total_loss
    token_counts["target"] = total_tokens
    return EvalBatchStats(metrics, loss_sums, token_counts)


def evaluate_single_stacked_batch(
    args: Args,
    model: GRCEGPT,
    dataset: TextDataset,
    split: str,
    block_length: int,
    batch_size: int,
    device: torch.device,
) -> EvalBatchStats:
    """Evaluate a stacked batch (cols per mode) with KV chaining."""

    metrics: dict[str, float | None] = {mode: None for mode in BATCH_MODES}
    metrics["target"] = None
    loss_sums = {mode: 0.0 for mode in BATCH_MODES}
    loss_sums["target"] = 0.0
    token_counts = {mode: 0 for mode in BATCH_MODES}
    token_counts["target"] = 0
    stacked_total_cols = effective_stacked_length(args, block_length)
    allocations = allocate_stacked_cols(
        stacked_total_cols,
        args.cols_encode,
        args.cols_decode,
        args.cols_forward,
        args.cols_noattn,
    )
    xb, yb = dataset.get_batch(split, stacked_total_cols, batch_size, device)
    embeddings = _sequence_embeddings(model, xb)
    cursor = 0
    kv_chain: list[list[tuple[torch.Tensor, torch.Tensor]] | None] = []
    grce_state = None
    xctx_state = None
    total_loss = 0.0
    total_tokens = 0
    for mode in BATCH_MODES:
        cols = allocations.get(mode, 0)
        if cols <= 0:
            continue
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
            loss_value = float(loss_sum.item())
            loss_sums[mode] = loss_value
            token_counts[mode] = token_count
            metrics[mode] = loss_value / token_count
            total_loss += loss_value
            total_tokens += token_count
        if mode != "noattn":
            kv_chain.append(kv_out)
        cursor += cols
    if total_tokens > 0:
        metrics["target"] = total_loss / total_tokens
    loss_sums["target"] = total_loss
    token_counts["target"] = total_tokens
    return EvalBatchStats(metrics, loss_sums, token_counts)


def evaluate_single_batch(
    args: Args,
    model: GRCEGPT,
    dataset: TextDataset,
    split: str,
    block_length: int,
    batch_size: int,
    device: torch.device,
    *,
    layout: str = "split",
) -> EvalBatchStats:
    if layout == "stacked":
        return evaluate_single_stacked_batch(
            args, model, dataset, split, block_length, batch_size, device
        )
    if layout == "split":
        return evaluate_single_split_batch(
            args, model, dataset, split, block_length, batch_size, device
        )
    if layout == "both":
        split_stats = evaluate_single_split_batch(
            args, model, dataset, split, block_length, batch_size, device
        )
        stacked_stats = evaluate_single_stacked_batch(
            args, model, dataset, split, block_length, batch_size, device
        )
        return _combine_eval_stats((split_stats, stacked_stats))
    raise ValueError(f"Unknown evaluation layout: {layout}")


def train_split_batch(
    args: Args,
    model: GRCEGPT,
    dataset: TextDataset,
    block_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    split_total_rows = effective_split_rows(args, batch_size)
    split_block_length = effective_split_length(args, block_length)
    allocations = allocate_split_rows(
        split_total_rows,
        args.rows_encode,
        args.rows_decode,
        args.rows_forward,
        args.rows_noattn,
    )
    total_loss_sum: torch.Tensor | None = None
    total_tokens = 0
    for mode in BATCH_MODES:
        rows = allocations.get(mode, 0)
        if rows <= 0:
            continue
        xb, yb = dataset.get_batch("train", split_block_length, rows, device)
        position_offsets = sample_position_offsets(
            rows,
            model.config.block_size,
            split_block_length,
            device,
        )
        logits, _, _ = model.forward_autoreg(
            xb,
            targets=yb,
            mode=mode,
            position_offsets=position_offsets,
        )
        loss_sum, token_count = loss_sum_and_token_count(logits, yb, last_only=(mode == "encode"))
        if token_count <= 0:
            continue
        total_loss_sum = loss_sum if total_loss_sum is None else total_loss_sum + loss_sum
        total_tokens += token_count
    if total_loss_sum is None:
        raise RuntimeError("No tokens processed in split batch")
    return total_loss_sum, total_tokens


def train_stacked_batch(
    args: Args,
    model: GRCEGPT,
    dataset: TextDataset,
    block_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    stacked_total_cols = effective_stacked_length(args, block_length)
    allocations = allocate_stacked_cols(
        stacked_total_cols,
        args.cols_encode,
        args.cols_decode,
        args.cols_forward,
        args.cols_noattn,
    )
    xb, yb = dataset.get_batch("train", stacked_total_cols, batch_size, device)
    embeddings = _sequence_embeddings(model, xb)
    cursor = 0
    kv_chain: list[list[tuple[torch.Tensor, torch.Tensor]] | None] = []
    grce_state = None
    xctx_state = None
    total_loss_sum: torch.Tensor | None = None
    total_tokens = 0
    for mode in BATCH_MODES:
        cols = allocations.get(mode, 0)
        if cols <= 0:
            continue
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
            total_loss_sum = loss_sum if total_loss_sum is None else total_loss_sum + loss_sum
            total_tokens += token_count
        if mode != "noattn":
            kv_chain.append(kv_out)
        cursor += cols
    if total_loss_sum is None:
        raise RuntimeError("No tokens processed in stacked batch")
    return total_loss_sum, total_tokens


def train_model(
    args: Args,
    model: GRCEGPT,
    dataset: TextDataset,
    device: torch.device,
    steps: int,
    block_length: int,
    batch_size: int,
    step_layouts: Sequence[str] | None,
    eval_interval: int,
    start_step: int,
    optimizer: torch.optim.Optimizer,
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
) -> Tuple[int, List[Dict[str, float]], float, float]:
    """Run the main training loop for a cycle."""

    if optimizer is None:
        raise ValueError("train_model requires an initialized optimizer instance")
    total_steps = start_step
    history_updates: List[Dict[str, float]] = []
    printed_header = False
    eval_interval = max(1, int(eval_interval))
    if step_layouts is None or len(step_layouts) != steps:
        step_layouts = build_cycle_layouts(steps, args.batch_layout, args.split_steps)
    else:
        step_layouts = list(step_layouts)

    if prompt_tracker is not None:
        prompt_queue = prompt_tracker.prompt_queue(reset=reset_prompt_queue)
    else:
        prompt_queue = []

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
    while step < steps:
        layout_choice = step_layouts[step]
        try:
            if layout_choice == "stacked":
                total_loss_sum, total_tokens = train_stacked_batch(
                    args,
                    model,
                    dataset,
                    block_length,
                    batch_size,
                    device,
                )
            elif layout_choice == "split":
                total_loss_sum, total_tokens = train_split_batch(
                    args,
                    model,
                    dataset,
                    block_length,
                    batch_size,
                    device,
                )
            elif layout_choice == "both":
                split_loss_sum, split_tokens = train_split_batch(
                    args,
                    model,
                    dataset,
                    block_length,
                    batch_size,
                    device,
                )
                stacked_loss_sum, stacked_tokens = train_stacked_batch(
                    args,
                    model,
                    dataset,
                    block_length,
                    batch_size,
                    device,
                )
                total_loss_sum = split_loss_sum + stacked_loss_sum
                total_tokens = split_tokens + stacked_tokens
            else:
                raise ValueError(f"Unknown batch layout: {layout_choice}")
            if total_tokens <= 0:
                raise RuntimeError("No tokens processed in training step")
            total_loss = total_loss_sum / float(total_tokens)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
        except torch.OutOfMemoryError:
            oom_retries += 1
            line_parts: List[str] = []
            if show_time:
                timestamp = time.strftime("%H:%M", time.localtime())
                line_parts.append(color_text(timestamp, Colors.BLUE))
            line_parts.append(color_text(f"{total_steps}", Colors.CYAN))
            line_parts.append(color_text(
                f"OOM (retry {oom_retries}/3) during {layout_choice} batch; "
                "refreshing layout and retrying", Colors.YELLOW))
            line = " | ".join(line_parts)
            print(line)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            if oom_retries >= 3:
                raise
            continue
        oom_retries = 0
        step += 1
        total_steps += 1

        eval_due = step == 1 or step % eval_interval == 0 or step == steps
        if not eval_due:
            continue
        eval_timer.start()
        model.eval()
        if total_steps % 2 == 1:
            eval_layout = "split"
        elif args.batch_layout == "split":
            eval_layout = "split"
        elif args.batch_layout == "stacked":
            eval_layout = "stacked"
        elif args.batch_layout == "both":
            eval_layout = "both"
        else:
            eval_layout = "stacked"
        layout_uses_split = eval_layout in {"split", "both"}
        layout_uses_stacked = eval_layout in {"stacked", "both"}
        eval_metrics: dict[str, EvalBatchStats] = {}
        with torch.no_grad():
            for split in ("train", "test"):
                eval_metrics[split] = evaluate_single_batch(
                    args,
                    model,
                    dataset,
                    split,
                    block_length,
                    batch_size,
                    device,
                    layout=eval_layout,
                )
        model.train()
        eval_timer.stop()

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

        prompt_text = tokenizer.decode_pretty(args, torch.tensor(prompt_ids))
        completion_text = tokenizer.decode_pretty(args, torch.tensor(completion_ids), alt=True)
        sample_prefix = color_text(prompt_text, Colors.CYAN)
        sample_suffix = color_text(completion_text, Colors.YELLOW)
        sample_render = (Colors.YELLOW if sampling_strategy == 'argmax' else Colors.CYAN) + \
                        f"{sampling_strategy}:{Colors.RESET} " + sample_prefix + sample_suffix

        def format_metric(dataset_split: str, key: str, use_colon: bool) -> str:
            value = eval_metrics[dataset_split].metrics.get(key)
            if key in ROW_METRIC_LOG_GROUP:
                sep = ": " if use_colon else "- "
            else:
                sep = ""
            if value is None:
                return f"{sep}****"
            return f"{sep}{value:.2f}"

        detail_keys = ROW_METRIC_LOG_KEYS

        def format_train_line() -> str:
            base = format_metric("train", "target", layout_uses_split)
            if not show_train_loss_details:
                return base
            diag = " ".join(
                format_metric("train", key, layout_uses_split) for key in detail_keys
            )
            return f"{base} {diag}"

        def format_test_line() -> str:
            base = format_metric("test", "target", layout_uses_split)
            if not show_test_loss_details:
                return base
            diag = " ".join(
                format_metric("test", key, layout_uses_split) for key in detail_keys
            )
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
            "train_loss": float(eval_metrics["train"].metrics.get("target", 0.0) or 0.0),
            "test_loss": float(eval_metrics["test"].metrics.get("target", 0.0) or 0.0),
            "train_wall_seconds": float(total_wall_seconds),
            "unix_time": float(eval_now),
            "train_cursor": int(dataset.positions.get("train", 0)),
            "test_cursor": int(dataset.positions.get("test", 0)),
            "step_split_eval": 0.0,
        }
        split_rows_eval = effective_split_rows(args, batch_size) if layout_uses_split else 0
        split_cols_eval = effective_split_length(args, block_length) if layout_uses_split else 0
        stacked_rows_eval = batch_size if layout_uses_stacked else 0
        stacked_cols_eval = (
            effective_stacked_length(args, block_length) if layout_uses_stacked else 0
        )
        split_tokens_est = split_rows_eval * split_cols_eval
        stacked_tokens_est = stacked_rows_eval * stacked_cols_eval
        total_tokens_est = split_tokens_est + stacked_tokens_est
        if total_tokens_est > 0:
            record["step_split_eval"] = split_tokens_est / total_tokens_est
        metric_keys = ["target"] + ROW_METRIC_HIST_KEYS
        for key in metric_keys:
            train_val = eval_metrics["train"].metrics.get(key)
            test_val = eval_metrics["test"].metrics.get(key)
            if train_val is not None:
                record[f"train_loss_{key}"] = float(train_val)
            if test_val is not None:
                record[f"test_loss_{key}"] = float(test_val)
        history_updates.append(record)

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

    split_len = effective_split_length(args, block_length)
    stacked_len = effective_stacked_length(args, block_length)
    split_rows = effective_split_rows(args, batch_size)
    max_len = max(block_length, split_len, stacked_len)
    max_rows = max(batch_size, split_rows)
    train_chars = (max_len + 1) * max_rows * 2
    dataset.prepare_cycle("train", train_chars)

    def train_step(tag: str) -> float:
        model.train()
        layout_choice = args.batch_layout
        if layout_choice == "random":
            layout_choice = random.choice(["split", "stacked"])
        if layout_choice == "stacked":
            total_loss_sum, total_tokens = train_stacked_batch(
                args,
                model,
                dataset,
                block_length,
                batch_size,
                device,
            )
        elif layout_choice == "split":
            total_loss_sum, total_tokens = train_split_batch(
                args,
                model,
                dataset,
                block_length,
                batch_size,
                device,
            )
        elif layout_choice == "both":
            split_loss_sum, split_tokens = train_split_batch(
                args,
                model,
                dataset,
                block_length,
                batch_size,
                device,
            )
            stacked_loss_sum, stacked_tokens = train_stacked_batch(
                args,
                model,
                dataset,
                block_length,
                batch_size,
                device,
            )
            total_loss_sum = split_loss_sum + stacked_loss_sum
            total_tokens = split_tokens + stacked_tokens
        else:
            raise ValueError(f"Unknown batch layout: {layout_choice}")
        if total_tokens <= 0:
            raise RuntimeError("No tokens processed during profiling step")
        loss = total_loss_sum / float(total_tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    warm_loss = train_step("warmup")
    print(color_text(f"Warm-up step loss: {warm_loss:.4f}", Colors.CYAN))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("profile_train_step"):
            prof_loss = train_step("profile")

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
    prefix = f"{args.corpus}_model_"
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
        self._train_tokens: torch.Tensor | None = None
        self._test_tokens: torch.Tensor | None = None
        self.newline_token_id: int | None = None
        self.boundary_blocklist: Sequence[int] | None = None
        self.default_prompt_boundary: bool = False
        self.model_path: pathlib.Path | None = None
        self.log_path: pathlib.Path | None = None
        self.tokenizer_json: str | None = None

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

    def _prepare_corpus(
        self,
    ) -> tuple[
        GPT2TokenizerWrapper,
        TextDataset,
        torch.Tensor,
        torch.Tensor,
        int | None,
        Sequence[int] | None,
        bool,
        str,
    ]:
        data_dir = pathlib.Path(self.args.data)
        train_path = data_dir / f"{self.args.corpus}-train.txt.gz"
        test_path = data_dir / f"{self.args.corpus}-test.txt.gz"
        model_dir = pathlib.Path(self.args.model)
        if not model_dir.exists():
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

        must_build_tokenizer = (
            self.args.command == "corpus" and getattr(self.args, "corpus_init", False)
        )
        train_cache_path = data_dir / f"{self.args.corpus}_tokens_train_{self.args.vocab_size}.pt"
        test_cache_path = data_dir / f"{self.args.corpus}_tokens_test_{self.args.vocab_size}.pt"
        need_corpus_for_create = False
        if self.args.command == "create":
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

        tokenizer_key = f"{self.args.corpus}_vocab_{self.args.vocab_size}"
        tokenizer_path = data_dir / f"{tokenizer_key}.json"
        if self.args.command == "create" and not tokenizer_path.exists():
            must_build_tokenizer = True
        tokenizer_json = getattr(self.args, "tokenizer_json_override", None)
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
        if self.args.vocab_size <= reserved_tokens:
            raise ValueError(
                f"--vocab-size must exceed reserved tokens ({reserved_tokens}); got {self.args.vocab_size}"
            )
        target_vocab = max(0, self.args.vocab_size - reserved_tokens)
        tokenizer = GPT2TokenizerWrapper(
            vocab_source,
            tokenizer_path,
            target_vocab,
            pretrained_json=tokenizer_json,
        )
        expected_vocab_size = self.args.vocab_size
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

        enforce_boundary_guard = not self.args.no_boundary
        boundary_blocklist = (
            tokenizer.leading_alpha_token_ids if enforce_boundary_guard else None
        )
        default_prompt_boundary = (
            enforce_boundary_guard
            and boundary_blocklist is not None
            and prompt_needs_boundary(self.args.prompt)
        )

        train_tokens, train_text, train_bytes = load_or_prepare_tokens(
            "train",
            train_path,
            full_train_text,
            train_cache_path,
            tokenizer,
            seed=1234,
        )

        test_tokens, test_text, test_bytes = load_or_prepare_tokens(
            "test",
            test_path,
            full_test_text,
            test_cache_path,
            tokenizer,
            seed=5678,
        )

        if train_text is None and train_bytes == 0:
            train_bytes = len(train_tokens)
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
            color_text(f"[tokenizer (wall/cpu/gpu)]", Colors.CYAN)
            + color_text(f" {tok_timer.stop()}\n", Colors.MAGENTA)
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

        self.tokenizer = tokenizer
        self.dataset = dataset
        self._train_tokens = train_tokens
        self._test_tokens = test_tokens
        self.newline_token_id = newline_token_id
        self.boundary_blocklist = boundary_blocklist
        self.default_prompt_boundary = default_prompt_boundary
        self.tokenizer_json = tokenizer_json
        return (
            tokenizer,
            dataset,
            train_tokens,
            test_tokens,
            newline_token_id,
            boundary_blocklist,
            default_prompt_boundary,
            tokenizer_json,
        )

    def cli_corpus(
        self,
        tokenizer: GPT2TokenizerWrapper,
        train_tokens: torch.Tensor,
        test_tokens: torch.Tensor,
    ) -> int:
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

        actions_done = False
        if getattr(self.args, "corpus_init", False):
            print(
                color_text(
                    "Tokenizer initialized and token caches updated; run 'train' to build a model.",
                    Colors.GREEN,
                )
            )
            actions_done = True
        if getattr(self.args, "corpus_print_train", None):
            emit_range("Train", train_tokens, self.args.corpus_print_train)
            actions_done = True
        if getattr(self.args, "corpus_print_test", None):
            emit_range("Test", test_tokens, self.args.corpus_print_test)
            actions_done = True
        if not actions_done:
            print(color_text("No corpus action selected", Colors.YELLOW))
        return 0

    def cli_prompts(
        self,
        tokenizer: GPT2TokenizerWrapper,
        model_path: pathlib.Path,
    ) -> int:
        target_path = Path(self.args.target) if self.args.target else model_path
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
        if getattr(self.args, "reset", False):
            entries = default_prompt_entries()
            statuses = [0] * len(entries)
            changed = True
        if getattr(self.args, "clear", False):
            entries = []
            statuses = []
            changed = True
        removes = sorted(set(getattr(self.args, "remove", [])), reverse=True)
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
        add_pair = getattr(self.args, "add", None)
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
        show_list = self.args.list or (
            not getattr(self.args, "reset", False)
            and not getattr(self.args, "clear", False)
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
        return 0

    def big_fat_old_main(self) -> int:
        """Dispatch the CLI command selected by :func:`grce_cli_args`.

        Handles corpus management, training/reporting flow, and subcommands such
        as ``size``. When running training it constructs the model/tokenizer and
        calls :func:`train_model`.
        """

        ansi_file = None
        try:
            orig_stdout, orig_stderr, log_file = sys.stdout, sys.stderr, None

            (
                tokenizer,
                dataset,
                train_tokens,
                test_tokens,
                newline_token_id,
                boundary_blocklist,
                default_prompt_boundary,
                tokenizer_json,
            ) = self._prepare_corpus()
            self.args.vocab_size = tokenizer.vocab_size
            model_dir = pathlib.Path(self.args.model)

            if self.args.command == "corpus":
                return self.cli_corpus(tokenizer, train_tokens, test_tokens)

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
                prefix = f"{self.args.corpus}_model_"
                model_path = model_dir / f"{prefix}{model_tag}.pt"
                log_path = model_dir / f"{prefix}{model_tag}.log"
                self.args.model_path_override = model_path
                self.args.log_path_override = log_path
            print(color_text(f"Model: {model_path}", Colors.CYAN))
            print(color_text(f"Logfile: {log_path}", Colors.BLUE))
            requires_checkpoint = self.args.command in {"train", "report", "test"}
            if self.args.command == "create" and model_path.exists():
                raise FileExistsError(
                    f"Checkpoint {model_path} already exists; delete it or pick a new --model directory."
                )
            if requires_checkpoint and not model_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint {model_path} not found; run 'create' first to initialize it."
                )

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
            prompt_tracker = PromptTracker(tokenizer)

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
                        if "dataset" in payload:
                            dataset.load_state(payload["dataset"])
                        total_steps = int(payload.get("total_steps", 0))
                        loss_history = list(payload.get("loss_history", []))
                        total_train_wall = float(payload.get("train_wall_seconds", 0.0))
                        prompt_tracker.load_state(payload.get("prompt_state"))
                        if self.args.checkpoint_optimizer:
                            optimizer_state = payload.get("optimizer")
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
                        "dataset": dataset.state_dict(),
                        "total_steps": total_steps,
                        "loss_history": loss_history,
                        "config": asdict(args_to_model_geometry(self.args)),
                        "train_wall_seconds": total_train_wall,
                        "prompt_state": prompt_tracker.serialize() if prompt_tracker else None,
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
            if self.args.command == "create":
                checkpoint_payload = {
                    "model": model.state_dict(),
                    "dataset": dataset.state_dict(),
                    "total_steps": 0,
                    "loss_history": [],
                    "config": asdict(args_to_model_geometry(self.args)),
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
                split_len = effective_split_length(self.args, self.args.block_length)
                stacked_len = effective_stacked_length(self.args, self.args.block_length)
                split_rows = effective_split_rows(self.args, self.args.batch_size)
                step_layouts = build_cycle_layouts(
                    self.args.steps,
                    self.args.batch_layout,
                    self.args.split_steps,
                )
                train_chars_cycle = 0
                for layout in step_layouts:
                    if layout == "split":
                        train_chars_cycle += (split_len + 1) * split_rows
                    elif layout == "stacked":
                        train_chars_cycle += (stacked_len + 1) * self.args.batch_size
                    elif layout == "both":
                        train_chars_cycle += (split_len + 1) * split_rows
                        train_chars_cycle += (stacked_len + 1) * self.args.batch_size
                    else:
                        raise ValueError(f"Unknown batch layout: {layout}")
                eval_calls = max(1, count_eval_calls(self.args.steps, self.args.eval_interval))
                split_eval_chars = (split_len + 1) * split_rows
                stacked_eval_chars = (stacked_len + 1) * self.args.batch_size
                if self.args.batch_layout == "split":
                    per_eval_chars = split_eval_chars
                elif self.args.batch_layout == "stacked":
                    per_eval_chars = stacked_eval_chars
                elif self.args.batch_layout == "both":
                    per_eval_chars = split_eval_chars + stacked_eval_chars
                else:
                    per_eval_chars = max(split_eval_chars, stacked_eval_chars)
                test_chars_cycle = per_eval_chars * eval_calls
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
                print(color_text(f"Model: {model_path}", Colors.CYAN))
                print(
                    color_text(
                        f"Corpus ranges: train tokens {train_range}, test tokens {test_range}",
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
                    step_layouts,
                    self.args.eval_interval,
                    total_steps,
                    optimizer,
                    prompt_tokens,
                    self.args.generate,
                    tokenizer,
                    suppress_newlines=self.args.no_newlines,
                    newline_token_id=newline_token_id,
                    prompt_tracker=prompt_tracker,
                    reset_prompt_queue=self.args.reset_prompt_each_cycle,
                    cycle_wall_start=cycle_wall,
                    base_wall_seconds=total_train_wall,
                    show_time=self.args.time,
                    default_prompt_boundary=default_prompt_boundary,
                    boundary_blocklist=boundary_blocklist,
                    show_train_loss_details=self.args.show_train_loss_details,
                    show_test_loss_details=self.args.show_test_loss_details,
                )
                loss_history.extend(updates)
                pure_train = Timer().add(train_timer).sub(eval_timer)
                acc_train.add(pure_train)
                acc_eval.add(eval_timer)
                total_train_wall += train_timer.wall_secs
                if not self.args.skip_model_update:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "dataset": dataset.state_dict(),
                            "total_steps": total_steps,
                            "loss_history": loss_history,
                            "config": asdict(args_to_model_geometry(self.args)),
                            "train_wall_seconds": total_train_wall,
                            "prompt_state": prompt_tracker.serialize() if prompt_tracker else None,
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
