"""GRCE proof-of-concept. Most of it is written by ChatGPT/Codex. I told
it to base it loosely the picoGPT.

This script keeps the picoGPT spirit of being small and hackable while
adding the Gradient-limited Recurrent Context Encoding (GRCE) channel described in
the README. It trains a tiny GPT-style tokenizer-backed Transformer on the bundled
Simple English Wikipedia split and shows how the recurrent context vector can
be integrated with a configurable gradient-limiting constraint across time.
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# GRCE Model Configuration
# -----------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 2000  # GPT-2 base supports ~50k merges; we stay small for the PoC.
    block_size: int = 64    # GPT-2 base uses 1024 tokens.
    n_layer: int = 12       # GPT-2 base uses 12 layers.
    n_head: int = 8         # GPT-2 base uses 12 attention heads.
    n_embd: int = 128       # GPT-2 base uses 768 embedding dims.
    n_grce: int = 64        # Narrow GRCE context dims.
    n_xctx: int = 384       # Wide XCTX context dims.
    dropout: float = 0.05
    detach_span: int = 0    # Detach gradients every N positions (0 disables detaching).
    detach_context: bool = True  # Whether to detach recurring context when span triggers.
    detach_layer: int = -1       # Layer index (1-based) after which to detach Transformer grads.


MODEL_CONFIG_TEMPLATE = ModelConfig()


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

def parse_range_arg(value: str) -> tuple[int, int]:
    parts = value.replace(" ", "").split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid range '{value}'. Expected format START-END.")
    start, end = int(parts[0]), int(parts[1])
    if end < start:
        raise ValueError(f"Range end {end} is smaller than start {start}.")
    return start, end

def compute_default_think_sequences(args: argparse.Namespace) -> int:
    context_channels_active = args.n_grce > 0 or args.n_xctx > 0
    special_rows = 0
    if context_channels_active:
        special_rows += 1  # pure Transformer row when context exists
    if args.n_xctx > 0:
        special_rows += 4  # no-XCTX, punctured XCTX, no-attn, punct-attn
    special_rows += 1  # random-think row
    available = args.batch_size - special_rows
    default_think = available // 2
    if default_think < 1:
        raise ValueError(
            "Default thinking requires at least two non-special rows; "
            f"batch-size {args.batch_size} minus {special_rows} special rows leaves {available}. "
            "Increase --batch-size, disable context dropout, or pass --no-think."
        )
    return default_think

def grce_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = MODEL_CONFIG_TEMPLATE
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
        default="simplerwiki",
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
        "--block-size",
        type=int,
        default=defaults.block_size,
        help="Number of tokens per training sample",
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
        help="Embedding/hidden dimension (GPT-2 base uses 768).",
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
            "Dimension of the wide (layer-partitioned) context channel; requires n_xctx to be divisible by n_layer"
        ),
    )
    model_group.add_argument(
        "--tokenizer-vocab",
        type=int,
        default=defaults.vocab_size,
        help="Total vocabulary size for the GPT-2 style tokenizer (including special tokens)",
    )
    model_group.add_argument(
        "--no-think",
        dest="disable_think",
        action="store_true",
        help=(
            "Disable thinking tokens entirely. By default the number of thinking rows is "
            "computed automatically from batch size and special-row requirements."
        ),
    )
    model_group.add_argument(
        "--undo",
        type=int,
        default=0,
        help="Enable undo pairs with up to N random+undo sequences per block",
    )
    model_group.add_argument(
        "--tiny",
        action="store_true",
        help=(
            "Shortcut for --batch-size 8 --block-size 8 --n-layer 3 --n-head 2 "
            "--n-embd 8 --n-grce 4 --n-xctx 12 --steps 2 --cycles 1"
        ),
    )

    training_group = parser.add_argument_group("Training schedule")
    training_group.add_argument("--steps", type=int, default=100, help="Training steps per cycle")
    training_group.add_argument(
        "--cycles",
        type=int,
        default=100,
        help="Repeat the full training/eval/update cycle N times.",
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of sequences per optimization step.",
    )
    training_group.add_argument(
        "--detach-span",
        type=int,
        default=0,
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
        "--context-dropout-interval",
        type=int,
        default=1,
        help=(
            "Every N steps create pure-Transformer, no-XCTX, punctured-XCTX, random-think, no-attention, "
            "and attention-punctured rows (0 disables)"
        ),
    )
    training_group.add_argument(
        "--reward-relu",
        type=float,
        default=0.0,
        help=(
            "Enable experimental ReLU reward updates with scale=10^{-value} (value<=0 disables)"
        ),
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
    training_group.add_argument(
        "--eval-iters",
        type=int,
        default=2,
        help="How many mini-batches to average for evaluation losses.",
    )
    training_group.add_argument(
        "--eval-full",
        type=int,
        default=-1,
        help=(
            "How often to run the expensive evaluation variants: default -1 runs them once per cycle; 0 "
            "disables them entirely; 1 runs them on every evaluation (the same cadence as --eval-interval); "
            "positive values >1 must be multiples of --eval-interval; negative values schedule evenly spaced "
            "full evals within each cycle (e.g., -1 means once at the end of a cycle, -2 means twice per cycle) "
            "and require --steps to be divisible by the absolute value so the cadence lines up."
        ),
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
        "--no-think-output",
        dest="no_think_output",
        action="store_true",
        help="During sampling/reporting, suppress thinking tokens entirely",
    )
    sampling_group.add_argument(
        "--no-think-prompt",
        action="store_true",
        help="Do not insert thinking tokens inside the prompt during sampling/reporting",
    )
    sampling_group.add_argument(
        "--think-hard",
        action="store_true",
        help="While processing the prompt, insert thinking tokens after every mispredicted token",
    )
    sampling_group.add_argument(
        "--no-boundary",
        action="store_true",
        help="Allow completions to continue immediately after the prompt without enforcing a word boundary",
    )
    sampling_group.add_argument(
        "--underline",
        action="store_true",
        help="Underline console tokens when the pronoun detector strongly expects a pronoun",
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
        "--train-loss-details",
        action="store_true",
        help="Show the special/noprev columns for train loss in the live log",
    )
    logging_group.add_argument(
        "--no-test-loss-details",
        action="store_true",
        help="Collapse the test loss group down to a single column in the live log",
    )
    logging_group.add_argument(
        "--long-loss-log",
        action="store_true",
        help="Always include the detailed loss columns in the live log even when data is missing",
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
        help="Print block_size tokens from the test corpus",
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

    args = parser.parse_args()
    raw_eval_full = getattr(args, "eval_full", None)
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
        if not flag_present("--block-size"):
            args.block_size = 8
        if not flag_present("--batch-size"):
            args.batch_size = 8
        if not flag_present("--n-layer"):
            args.n_layer = 3
        if not flag_present("--n-head"):
            args.n_head = 2
        if not flag_present("--n-embd"):
            args.n_embd = 8
        if not flag_present("--n-grce"):
            args.n_grce = 4
        if not flag_present("--n-xctx"):
            args.n_xctx = 12
        if not flag_present("--steps"):
            args.steps = 2
        if not flag_present("--cycles"):
            args.cycles = 1
        if not flag_present("--eval-interval"):
            args.eval_interval = 1
        if not flag_present("--corpus"):
            args.corpus = "simplestwiki"
    if raw_eval_full is not None:
        try:
            eval_full_value = int(raw_eval_full)
        except (TypeError, ValueError):
            parser.error("--eval-full must be an integer")
        eval_full = eval_full_value
        if eval_full < 0:
            if args.steps <= 0:
                parser.error("--eval-full negative values require --steps > 0")
            offset = abs(eval_full)
            if args.steps % offset != 0:
                parser.error("--eval-full -N requires --steps to be divisible by N")
            eval_full = args.steps // offset
        eval_full = max(0, eval_full)
        if eval_full == 0:
            if getattr(args, "train_loss_details", False):
                parser.error("--eval-full 0 cannot be combined with --train-loss-details")
            args.no_test_loss_details = True
        elif eval_full != 1:
            interval = max(1, args.eval_interval)
            if eval_full % interval != 0:
                parser.error("--eval-full must be 0, 1, or a multiple of --eval-interval")
        args.eval_full = eval_full
    if getattr(args, "disable_think", False):
        args.think = 0
    else:
        args.think = compute_default_think_sequences(args)
    return args

if __name__ == "__main__":
    cli_args = grce_cli_args(sys.argv)


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

# Local GPT2 tokenizer adapter (no huggingface dependency)
class GPT2TokenizerFast:
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

FANCY_SPACE = "\u2423"  # Open Box symbol for visible spaces
FANCY_ENTER = "\u23CE"  # Return symbol for visible newlines
THINK_TOKEN = "<think>"
THINK_SYMBOL = "\u2754"  # white question mark
# THINK_SYMBOL = "\u21BA"  # anticlockwise circle arrow (alternative option)
UNDO_TOKEN = "<undo>"
UNDO_SYMBOL = "\u21A9"  # leftwards arrow with hook
ASCII_LETTERS = set(string.ascii_letters)
ASCII_LOWERCASE = set(string.ascii_lowercase)

PROMPT_GOALS = [
    ("one plus one is", " two"),
    ("fire is hot and ice is", " cold"),
    ("the opposite of up is", " down"),
    ("ice is cold and  fire is", " hot"),
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
    UNDERLINE = "\033[4m"
    UNDERLINE = "\033[4m"
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


NOUN_PROFORM_WORDS = [
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "mine",
    "yours",
    "hers",
    "ours",
    "theirs",
    "myself",
    "yourself",
    "himself",
    "herself",
    "itself",
    "ourselves",
    "yourselves",
    "themselves",
    "this",
    "that",
    "these",
    "those",
    "who",
    "whom",
    "whose",
]

NON_NOUN_PROFORM_WORDS = [
    "do",
    "does",
    "did",
    "done",
    "doing",
    "so",
    "such",
    "thus",
    "there",
    "here",
    "then",
    "therefore",
    "thereby",
    "therein",
    "thereof",
]

PRONOUN_DOMINANCE_RATIO = 2.0
PRONOUN_MIN_MASS = 0.05


class NounExpectationDetector:
    def __init__(
        self,
        model: "GRCEGPT",
        tokenizer: "GPT2TokenizerWrapper",
        think_token_id: int | None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.think_token_id = think_token_id
        self.max_context = getattr(model.config, "block_size", 0)
        self.noun_token_ids = sorted(tokenizer.noun_proform_token_ids)
        self.other_token_ids = sorted(tokenizer.other_proform_token_ids)

    def _prob_mass(self, probs: torch.Tensor, token_ids: list[int]) -> float:
        if not token_ids:
            return 0.0
        index = torch.tensor(token_ids, device=probs.device, dtype=torch.long)
        return float(probs.index_select(0, index).sum().item())

    def requires_pronoun(self, prefix_tokens: list[int]) -> bool:
        if not prefix_tokens or not self.noun_token_ids:
            return False
        context = prefix_tokens[-self.max_context :] if self.max_context > 0 else prefix_tokens
        idx = torch.tensor(context, dtype=torch.long, device=self.device).unsqueeze(0)
        was_training = self.model.training
        if was_training:
            self.model.eval()
        try:
            with torch.no_grad():
                logits, _, _ = self.model.forward_autoreg(
                    idx,
                    think_token_id=self.think_token_id,
                )
        finally:
            if was_training:
                self.model.train()
        next_logits = logits[:, -1, :].squeeze(0)
        probs = torch.softmax(next_logits, dim=-1)
        pronoun_mass = self._prob_mass(probs, self.noun_token_ids)
        if pronoun_mass < PRONOUN_MIN_MASS:
            return False
        other_mass = self._prob_mass(probs, self.other_token_ids)
        if other_mass <= 0.0:
            return True
        return pronoun_mass >= other_mass * PRONOUN_DOMINANCE_RATIO


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


class GPT2TokenizerWrapper:
    EXTRA_SPECIAL_TOKENS = [THINK_TOKEN, UNDO_TOKEN]

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
        self.think_id = self.tokenizer.convert_tokens_to_ids(THINK_TOKEN)
        if self.think_id is None:
            raise ValueError("Failed to add think token to tokenizer vocabulary")
        self.undo_id = self.tokenizer.convert_tokens_to_ids(UNDO_TOKEN)
        if self.undo_id is None:
            raise ValueError("Failed to add undo token to tokenizer vocabulary")
        self.non_special_ids = [
            tok_id for tok_id in range(self.vocab_size) if tok_id not in self.special_ids
        ]
        self.noun_proform_token_ids = self._collect_proform_token_ids(NOUN_PROFORM_WORDS)
        self.other_proform_token_ids = self._collect_proform_token_ids(
            NON_NOUN_PROFORM_WORDS
        )
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

    def _collect_proform_token_ids(self, words: list[str]) -> set[int]:
        token_ids: set[int] = set()
        for word in words:
            token_ids.update(self._token_ids_for_word(word))
        return token_ids

    def _token_ids_for_word(self, word: str) -> set[int]:
        base = word.lower()
        token_ids: set[int] = set()
        variants = {base, base.capitalize(), base.upper()}
        for variant in variants:
            for prefix in ("", " "):
                text = f"{prefix}{variant}"
                encoded = self.tokenizer.encode(text, add_special_tokens=False)
                if len(encoded) != 1:
                    continue
                tok_id = encoded[0]
                decoded = (
                    self.tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
                    .strip()
                    .lower()
                )
                if decoded == base:
                    token_ids.add(tok_id)
        return token_ids

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

    def is_noun_proform_token(self, token_id: int) -> bool:
        return token_id in self.noun_proform_token_ids

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
        self.prompts = prompt_entries if prompt_entries else default_prompt_entries()
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
        block_size: int,
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
        span = block_size + 1
        if len(chunk) <= span:
            raise ValueError(
                f"Chunk for {split} must be larger than block size ({len(chunk)} <= {span})."
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


def compute_prefill_contexts(
    model: GRCEGPT,
    prefill_tokens: torch.Tensor,
    *,
    think_token_id: int | None,
) -> list[torch.Tensor] | None:
    if prefill_tokens.numel() == 0:
        return None
    if not model.context_channels:
        return None
    with torch.no_grad():
        _, _, extras = model.forward_autoreg(
            prefill_tokens,
            think_token_id=think_token_id,
            record_final_context=True,
        )
    if not extras:
        return None
    final_contexts = extras.get("final_context_raw")
    if not final_contexts:
        return None
    return [ctx.detach() if ctx is not None else None for ctx in final_contexts]


@dataclass
class SpecialRowMasks:
    context_special_rows: set[int]
    context_disabled_mask: torch.Tensor | None
    xctx_disabled_mask: torch.Tensor | None
    context_dropout_positions: torch.Tensor | None
    attention_disabled_mask: torch.Tensor | None
    attention_dropout_positions: torch.Tensor | None
    think_disabled_rows: set[int]
    forced_think_rows: set[int]


def build_special_row_masks(
    batch_size: int,
    block_size: int,
    device: torch.device,
    *,
    think_enabled: bool,
    context_enabled: bool,
    xctx_enabled: bool,
) -> SpecialRowMasks:
    context_special_rows: set[int] = set()
    think_disabled_rows: set[int] = set()
    forced_think_rows: set[int] = set()
    context_disabled_mask: torch.Tensor | None = None
    xctx_disabled_mask: torch.Tensor | None = None
    context_dropout_positions: torch.Tensor | None = None
    attention_disabled_mask: torch.Tensor | None = None
    attention_dropout_positions: torch.Tensor | None = None
    if batch_size <= 0:
        return SpecialRowMasks(
            context_special_rows,
            context_disabled_mask,
            xctx_disabled_mask,
            context_dropout_positions,
            attention_disabled_mask,
            attention_dropout_positions,
            think_disabled_rows,
            forced_think_rows,
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

    if think_enabled:
        think_disabled_rows.add(random.randrange(batch_size))
        candidates = [idx for idx in range(batch_size) if idx not in think_disabled_rows]
        if candidates:
            forced_idx = pick_row(occupied_rows, disallowed=think_disabled_rows)
            forced_think_rows.add(forced_idx)

    return SpecialRowMasks(
        context_special_rows,
        context_disabled_mask,
        xctx_disabled_mask,
        context_dropout_positions,
        attention_disabled_mask,
        attention_dropout_positions,
        think_disabled_rows,
        forced_think_rows,
    )


@dataclass
class ThinkSettings:
    max_steps: int = 0
    token_id: int | None = None

    @property
    def enabled(self) -> bool:
        return self.max_steps > 0 and self.token_id is not None


def active_think_token_id(think: ThinkSettings | None) -> int | None:
    if think is None or not think.enabled or think.token_id is None:
        return None
    return int(think.token_id)
@dataclass
class UndoSettings:
    max_pairs: int = 0
    token_id: int | None = None
    fill_choices: list[int] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return (
            self.max_pairs > 0
            and self.token_id is not None
            and bool(self.fill_choices)
        )


class ReLURewardTracker:
    def __init__(
        self,
        model: GRCEGPT,
        steps: int,
        *,
        ratio: float = 0.2,
        scale: float = 1e-4,
    ) -> None:
        device = next(model.parameters()).device
        hidden = 4 * model.config.n_embd
        self.good_counts = [torch.zeros(hidden, device=device) for _ in range(model.config.n_layer)]
        self.bad_counts = [torch.zeros(hidden, device=device) for _ in range(model.config.n_layer)]
        self.interval = max(1, steps // 2)
        self.ratio = ratio
        self.token_ratio = 0.25
        self.scale = max(0.0, float(scale))
        self.model = model
        self.pending_steps = 0

    def record_batch(
        self,
        relu_activity: list[list[torch.Tensor | None]] | None,
        token_losses: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if relu_activity is None:
            return
        good_scores, bad_scores = self._compute_token_scores(token_losses, valid_mask)
        for t, per_layer in enumerate(relu_activity):
            if per_layer is None:
                continue
            good_weight = good_scores[:, t].unsqueeze(1)
            bad_weight = bad_scores[:, t].unsqueeze(1)
            good_active = good_weight.any()
            bad_active = bad_weight.any()
            if not good_active and not bad_active:
                continue
            for layer_idx, mask in enumerate(per_layer):
                if mask is None:
                    continue
                if good_active:
                    contrib = (mask.float() * good_weight).sum(dim=0)
                    self.good_counts[layer_idx] += contrib
                if bad_active:
                    inactive = (~mask).float()
                    contrib = (inactive * bad_weight).sum(dim=0)
                    self.bad_counts[layer_idx] += contrib
        self.pending_steps += 1

    def maybe_apply(self) -> None:
        if self.pending_steps >= self.interval:
            self.apply_updates()

    def finalize(self) -> None:
        if any(count.sum().item() != 0 for count in self.good_counts + self.bad_counts):
            self.apply_updates()

    def apply_updates(self) -> None:
        if self.scale <= 0:
            self.pending_steps = 0
            return
        self.pending_steps = 0
        for layer_idx, block in enumerate(self.model.core.blocks):
            combined = self.good_counts[layer_idx] + self.bad_counts[layer_idx]
            self._apply_bias(block.ff.fc1.bias, combined)
            self.good_counts[layer_idx].zero_()
            self.bad_counts[layer_idx].zero_()
        print(
            color_text(
                f"[reward-relu] applied auxiliary updates (scale={self.scale:.2e})",
                Colors.MAGENTA,
            )
        )

    def _apply_bias(self, bias: torch.Tensor, scores: torch.Tensor) -> None:
        if bias is None or scores.numel() == 0:
            return
        positive = scores > 0
        if not positive.any():
            return
        hidden = bias.size(0)
        k = max(1, int(hidden * self.ratio))
        available = positive.sum().item()
        k = min(k, available)
        if k <= 0:
            return
        values, indices = torch.topk(scores, k)
        bias_std = bias.data.std().item()
        delta = self.scale * bias_std
        if delta == 0:
            return
        bias.data[indices] += delta

    def _compute_token_scores(
        self, token_losses: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        good = torch.zeros_like(token_losses)
        bad = torch.zeros_like(token_losses)
        B, T = token_losses.shape
        for b in range(B):
            valid_indices = torch.nonzero(valid_mask[b], as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            losses = token_losses[b, valid_indices]
            slice_size = max(1, int(valid_indices.numel() * self.token_ratio))
            slice_size = min(slice_size, valid_indices.numel())
            if slice_size <= 0:
                continue
            mean_loss = losses.mean()
            good_vals, good_pos = torch.topk(losses, slice_size, largest=False)
            bad_vals, bad_pos = torch.topk(losses, slice_size, largest=True)
            good_scores = (mean_loss - good_vals).clamp(min=0)
            bad_scores = (bad_vals - mean_loss).clamp(min=0)
            good_indices = valid_indices[good_pos]
            bad_indices = valid_indices[bad_pos]
            good[b, good_indices] = good_scores
            bad[b, bad_indices] = bad_scores
        return good, bad
def augment_training_batch(
    model: GRCEGPT,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    think: ThinkSettings | None,
    undo: UndoSettings | None,
    *,
    disable_context_rows: set[int] | None = None,
    disable_think_rows: set[int] | None = None,
    forced_think_rows: set[int] | None = None,
    initial_context_raw: list[torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    think_enabled = think is not None and think.enabled
    think_token_id = active_think_token_id(think)
    forced_context_off = disable_context_rows or set()
    forced_think_rows = forced_think_rows or set()
    think_disabled_rows = disable_think_rows or set()
    undo_enabled = undo is not None and undo.enabled
    if not think_enabled and not undo_enabled:
        return inputs, targets, None, None, None
    B, block_size = inputs.shape
    device = inputs.device
    new_inputs = inputs.clone()
    new_targets = targets.clone()
    random_mask = None
    think_labels = None
    think_slot_mask = None
    if undo_enabled:
        random_mask = torch.zeros_like(inputs, dtype=torch.bool, device=device)
    full_logits = None
    prev_mode = model.training
    need_logits = think_enabled or undo_enabled
    if need_logits:
        model.eval()
        with torch.no_grad():
            full_logits, _, _ = model.forward_autoreg(
                inputs,
                think_token_id=think_token_id,
                initial_context_raw=initial_context_raw,
            )
    if prev_mode and need_logits:
        model.train()
    think_scores = None
    if think_enabled:
        think_labels = torch.full_like(inputs, -1)
        think_slot_mask = torch.zeros_like(inputs, dtype=torch.bool, device=device)
        if full_logits is not None and think is not None and think.token_id is not None:
            think_scores = full_logits[..., think.token_id]
    seq_think_quota = 0
    think_row_set: set[int] = set()
    special_think_rows: set[int] = set(forced_think_rows)
    if think_enabled and think is not None:
        seq_think_quota = max(0, int(think.max_steps))
        if seq_think_quota > 0:
            eligible_rows = [
                row_idx
                for row_idx in range(B)
                if row_idx not in think_disabled_rows
                and row_idx not in forced_context_off
                and row_idx not in forced_think_rows
            ]
            quota = min(seq_think_quota, len(eligible_rows))
            think_row_set = set(eligible_rows[:quota])
        special_think_rows.update(forced_think_rows)

    def sample_scaled_value() -> int:
        u = random.random()
        return int((u * u * block_size) / 4)

    for row in range(B):
        row_is_special = row in special_think_rows
        row_think_active = False
        if think_enabled and think is not None and think.token_id is not None:
            if row_is_special:
                row_think_active = True
            elif row in think_row_set:
                row_think_active = True
        max_insert_budget = block_size - 1
        undo_cap = undo.max_pairs if undo_enabled else 0
        remaining_budget = max_insert_budget
        undo_cap = min(undo_cap, remaining_budget // 2)
        undo_pairs = random.randint(0, undo_cap) if undo_cap > 0 else 0
        keep_len = block_size - 2 * undo_pairs
        keep_len = max(1, keep_len)
        seq_entries = [
            {
                "token": int(inputs[row, idx].item()),
                "tag": "base",
                "base_index": idx,
            }
            for idx in range(keep_len)
        ]
        tail_token = int(targets[row, keep_len - 1].item())
        tail_entry = {"token": tail_token, "tag": "base", "base_index": None}

        def truncate_entries() -> None:
            limit = block_size + 1
            if len(seq_entries) > limit:
                del seq_entries[limit:]

        if undo_enabled and undo_pairs > 0:
            row_logits = full_logits[row] if full_logits is not None else None
            for _ in range(undo_pairs):
                insert_limit = max(0, len(seq_entries) - 1)
                insert_pos = random.randint(0, insert_limit)
                base_idx = seq_entries[insert_pos]["base_index"]
                if row_logits is None:
                    raise ValueError("Undo logits unavailable during augmentation")
                prob_vec = row_logits[-1] if base_idx is None else row_logits[base_idx]
                probs = F.softmax(prob_vec, dim=-1)
                true_token = int(seq_entries[insert_pos]["token"])
                probs = probs.clone()
                probs[true_token] = 0.0
                probs = probs / probs.sum()
                filler = int(torch.multinomial(probs, num_samples=1).item())
                insert_limit = max(0, len(seq_entries) - 1)
                insert_pos = random.randint(0, insert_limit)
                seq_entries.insert(
                    insert_pos,
                    {"token": filler, "tag": "undo_filler", "base_index": None},
                )
                seq_entries.insert(
                    insert_pos + 1,
                    {
                        "token": int(undo.token_id),
                        "tag": "undo_marker",
                        "base_index": None,
                    },
                )
        truncate_entries()

        think_token_val = int(think.token_id) if think is not None and think.token_id is not None else None
        if row_think_active and think_token_val is not None:
            if row_is_special:
                max_k = max(0, (3 * block_size) // 4)
                k_val = random.randint(0, max_k)
                prob = k_val / max(1, block_size)
                expanded: list[dict] = []
                for idx_entry, entry in enumerate(seq_entries):
                    if idx_entry > 0 and random.random() < prob:
                        expanded.append(
                            {
                                "token": think_token_val,
                                "tag": "think_random",
                                "base_index": None,
                            }
                        )
                    expanded.append(entry)
                seq_entries = expanded
                truncate_entries()
            else:
                R = sample_scaled_value()
                H = sample_scaled_value()
                T = sample_scaled_value()
                if R > 0:
                    expanded: list[dict] = []
                    for entry in seq_entries:
                        expanded.append(entry)
                        for _ in range(R):
                            expanded.append(
                                {
                                    "token": think_token_val,
                                    "tag": "think_repeat",
                                    "base_index": None,
                                }
                            )
                    seq_entries = expanded
                    truncate_entries()
                if T > 0 and think_scores is not None:
                    for _ in range(T):
                        base_positions = [
                            (idx, entry)
                            for idx, entry in enumerate(seq_entries)
                            if entry.get("base_index") is not None
                        ]
                        if not base_positions:
                            break
                        scores = [
                            float(think_scores[row, entry[1]["base_index"]].item())
                            for entry in base_positions
                        ]
                        max_score = max(scores)
                        weights = [math.exp(val - max_score) for val in scores]
                        total_weight = sum(weights)
                        if total_weight <= 0:
                            break
                        normalized = [w / total_weight for w in weights]
                        chosen_entry = random.choices(base_positions, weights=normalized, k=1)[0]
                        insert_idx = chosen_entry[0] + 1
                        seq_entries.insert(
                            insert_idx,
                            {
                                "token": think_token_val,
                                "tag": "think_weighted",
                                "base_index": None,
                            },
                        )
                        truncate_entries()
                if H > 0:
                    for _ in range(H):
                        if len(seq_entries) >= block_size:
                            seq_entries.pop()
                        last_base = None
                        for idx in range(len(seq_entries) - 1, -1, -1):
                            if seq_entries[idx].get("base_index") is not None:
                                last_base = idx
                                break
                        insert_idx = (last_base + 1) if last_base is not None else len(seq_entries)
                        seq_entries.insert(
                            insert_idx,
                            {
                                "token": think_token_val,
                                "tag": "think_tail",
                                "base_index": None,
                            },
                        )
                        truncate_entries()

        seq_entries.append(tail_entry)
        truncate_entries()
        if len(seq_entries) < block_size + 1:
            padding_token = seq_entries[-1]["token"]
            while len(seq_entries) < block_size + 1:
                seq_entries.append({"token": padding_token, "tag": "pad", "base_index": None})

        seq_tensor = torch.tensor(
            [entry["token"] for entry in seq_entries], dtype=inputs.dtype, device=device
        )
        new_inputs[row] = seq_tensor[:-1]
        new_targets[row] = seq_tensor[1:]
        if random_mask is not None:
            for idx, entry in enumerate(seq_entries[:-1]):
                if entry.get("tag") == "undo_filler":
                    target_idx = idx - 1
                    if 0 <= target_idx < block_size:
                        random_mask[row, target_idx] = True
        if think_labels is not None and think_token_val is not None:
            for idx in range(block_size):
                if int(new_inputs[row, idx].item()) == think_token_val:
                    think_labels[row, idx] = int(new_targets[row, idx].item())
        if think_slot_mask is not None:
            think_slot_mask[row].fill_(row_think_active)

    return new_inputs, new_targets, random_mask, think_labels, think_slot_mask



def apply_think_slot_mask(
    logits: torch.Tensor,
    think_slot_mask: torch.Tensor | None,
    think: ThinkSettings | None,
) -> torch.Tensor:
    if (
        think_slot_mask is None
        or think is None
        or not think.enabled
        or think.token_id is None
    ):
        return logits
    disable_mask = (~think_slot_mask).to(device=logits.device)
    if not disable_mask.any():
        return logits
    logits[..., think.token_id] = logits[..., think.token_id].masked_fill(
        disable_mask, -1e9
    )
    return logits


def disable_think_logits(
    logits: torch.Tensor, think: ThinkSettings | None
) -> torch.Tensor:
    if think is None or not think.enabled or think.token_id is None:
        return logits
    logits[..., int(think.token_id)] = -1e9
    return logits


def compute_think_alignment_loss(
    model: GRCEGPT,
    logits: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    per_token_loss: torch.Tensor,
    think: ThinkSettings | None,
) -> torch.Tensor:
    if think is None or not think.enabled or think.token_id is None:
        return logits.new_tensor(0.0)
    think_id = int(think.token_id)
    mask = inputs == think_id
    if not mask.any():
        return logits.new_tensor(0.0)
    emb_table = model.core.tok_emb.weight
    think_emb = emb_table[think_id]
    head = model.core.head
    loss_accum = logits.new_tensor(0.0)
    count = 0
    B, T = inputs.shape
    for b in range(B):
        for t in range(T - 1):
            if not mask[b, t]:
                continue
            target_id = int(targets[b, t].item())
            next_target = int(targets[b, t + 1].item())
            if target_id == LOSS_IGNORE_INDEX or next_target == LOSS_IGNORE_INDEX:
                continue
            A = float(per_token_loss[b, t].item())
            B_loss = float(per_token_loss[b, t + 1].item())
            denom = A + B_loss
            if denom <= 0:
                continue
            ratio = A / (denom + 1e-6)
            next_emb = emb_table[target_id]
            mix_vec = next_emb + think_emb * ratio
            mix_vec = next_emb + 0.5 * (mix_vec - next_emb)
            target_logits = head(mix_vec.unsqueeze(0)).squeeze(0)
            target_probs = F.softmax(target_logits.detach(), dim=-1)
            output_logits = logits[b, t]
            log_probs = F.log_softmax(output_logits, dim=-1)
            diff = target_probs - log_probs.exp()
            loss_accum = loss_accum + 0.5 * torch.sum(diff * diff)
            count += 1
    if count > 0:
        loss_accum = loss_accum / count
    return loss_accum



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


def _build_geometry(config: ModelConfig, block_size: int) -> list[tuple[str, str, int]]:
    return [
        ("V", "vocab size", config.vocab_size),
        ("B", "block size", block_size),
        ("L", "transformer layers", config.n_layer),
        ("H", "attention heads", config.n_head),
        ("E", "embedding width", config.n_embd),
        ("G", "grce width", config.n_grce),
        ("X", "xctx width", config.n_xctx),
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
        {"label": "tokens", "count": V * E, "formula": "V * E"},
        {"label": "positions", "count": B * E, "formula": "B * E"},
    ]
    sections.append(("global", "Global resources", global_items))

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
    sections.append(("transformer", "Transformer resources", transformer_items))

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
    sections.append(("grce", "GRCE channel", grce_items))

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
    sections.append(("xctx", "XCTX channel", xctx_items))

    # Placeholder for summary, filled later
    return sections


def _append_summary_section(sections: list[tuple[str, str, list[dict]]]) -> list[tuple[str, str, list[dict]]]:
    totals: dict[str, int] = {}
    for key, _title, items in sections:
        totals[key] = sum(item["count"] for item in items)
    summary_items = [
        {"label": "global", "count": totals.get("global", 0), "formula": ""},
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
    sections.append(("summary", "Model parameter breakdown", summary_items))
    return sections


def _print_geometry(geometry: list[tuple[str, str, int]]) -> None:
    print(color_text("Model geometry:", Colors.CYAN, bold=True))
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
        line = f"  {label:<18} {count:>15,}"
        if formula:
            line += f"  ({formula})"
        print(line)
    print(f"  {'total':<18} {total:>15,}")
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

    counts[("global", "tokens")] = _module_param_count(model.core.tok_emb)
    counts[("global", "positions")] = _module_param_count(model.core.pos_emb)

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
        sampler_count = _module_list_param_count(channel.pre_norms) + _module_list_param_count(
            channel.context_sampler
        )
        mlp_count = (
            _module_param_count(channel.context_fuse_norm)
            + _module_param_count(channel.context_mlp)
            + _module_param_count(channel.context_norm)
        )
        bias_count = _module_list_param_count(channel.context_bias_gen)
        counts[(key, "samplers")] = counts.get((key, "samplers"), 0) + sampler_count
        counts[(key, "mlp")] = counts.get((key, "mlp"), 0) + mlp_count
        counts[(key, "bias")] = counts.get((key, "bias"), 0) + bias_count
    return counts


def _dominant_estimates(config: ModelConfig) -> list[tuple[str, int, str]]:
    L = config.n_layer
    E = config.n_embd
    G = config.n_grce
    X = config.n_xctx
    estimates = [
        ("transformer", 12 * L * E * E, "12 * n_layer * n_embd^2"),
    ]
    if G > 0:
        estimates.append(
            (
                "grce",
                2 * L * E * G + 8 * G * G,
                "2 * n_layer * n_embd * n_grce + 8*n_grce^2",
            )
        )
    if X > 0:
        estimates.append(
            (
                "xctx",
                2 * E * X + 2 * X * X + (8 * X * X) // max(1, L),
                "2 * n_embd * n_xctx + 2*n_xctx^2 + 8*n_xctx^2 / n_layer",
            )
        )
    return estimates


def describe_model_size(
    config: ModelConfig,
    block_size: int,
    *,
    check: bool = False,
    estimate: bool = False,
) -> None:
    geometry = _build_geometry(config, block_size)
    sections = _append_summary_section(_expected_sections(config, block_size))
    _print_geometry(geometry)
    for idx, (key, title, items) in enumerate(sections):
        print()
        if key == "summary":
            print(color_text(title, Colors.CYAN, bold=True))
            for entry in items:
                label = entry["label"]
                count = entry["count"]
                line = f"  {label:<18} {count:>15,}"
                print(line)
            continue
        _print_section(title, items)

    if estimate:
        print()
        print(color_text("Estimate using dominant terms", Colors.CYAN, bold=True))
        for label, count, formula in _dominant_estimates(config):
            print(f"  {label:<12} {count:>15,}  ({formula})")

    if check:
        expected_map = _flatten_expected(sections)
        actual_map = _compute_actual_counts(config)
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out = self.attn(
            self.ln1(x),
            disable_rows=attention_disabled_rows,
            dropout_positions=attention_dropout_positions,
        )
        if attention_disabled_rows is not None and attention_disabled_rows.any():
            mask = (~attention_disabled_rows).view(-1, 1, 1).to(attn_out.dtype)
            attn_out = attn_out * mask
        x = x + attn_out
        pre_ff = self.ln2(x)
        ff_out, mask = self.ff(pre_ff, record_mask=record_mask)
        x = x + ff_out
        return x, mask


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

    def forward(
        self,
        idx: torch.Tensor,
        block_biases: List[torch.Tensor] | None = None,
        *,
        record_relu_mask: bool = False,
        attention_disabled_rows: torch.Tensor | None = None,
        attention_dropout_positions: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor] | None]:
        B, T = idx.shape
        device = idx.device
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
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
        for layer_idx, block in enumerate(self.blocks):
            if block_biases is not None:
                x = x + block_biases[layer_idx]
            block_inputs.append(x[:, -1, :])
            x, layer_mask = block(
                x,
                record_mask=record_relu_mask,
                attention_disabled_rows=attention_disabled_rows,
                attention_dropout_positions=attention_dropout_positions,
            )
            if self.detach_layer > 0 and (layer_idx + 1) == self.detach_layer:
                x = x.detach()
            if record_relu_mask and relu_masks is not None and layer_mask is not None:
                relu_masks[layer_idx] = layer_mask
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, x, block_inputs, relu_masks


class GRCEContextChannel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        width: int,
        is_xctx: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.context_dim = int(width)
        self.is_xctx = is_xctx
        self.disabled = self.context_dim <= 0
        self.detach_span = max(0, int(getattr(config, "detach_span", 0)))
        self.detach_context = bool(getattr(config, "detach_context", True))
        if self.is_xctx:
            if config.n_layer <= 0 or self.context_dim % config.n_layer != 0:
                raise ValueError("--n-xctx requires n_xctx to be divisible by n_layer")
            self.layer_chunk = self.context_dim // config.n_layer
        else:
            self.layer_chunk = None
        if not self.disabled:
            self.pre_norms = nn.ModuleList(
                nn.LayerNorm(config.n_embd) for _ in range(config.n_layer)
            )
            self.context_sampler = nn.ModuleList(
                [self._build_sampler(config.n_embd, self.context_dim) for _ in range(config.n_layer)]
            )
            if self.is_xctx:
                layers = max(1, config.n_layer)
                mid = max(1, (4 * self.context_dim) // layers)
            else:
                mid = 4 * self.context_dim
            self.context_fuse_norm = nn.LayerNorm(self.context_dim)
            self.context_mlp = nn.Sequential(
                nn.Linear(self.context_dim, mid),
                nn.ReLU(),
                nn.Linear(mid, self.context_dim),
            )
            self.context_norm = nn.LayerNorm(self.context_dim)
            self.context_bias_gen = nn.ModuleList(
                [self._build_bias(self.context_dim, config.n_embd) for _ in range(config.n_layer)]
            )

    def _build_sampler(self, in_dim: int, out_dim: int) -> nn.Module:
        if not self.is_xctx or self.layer_chunk is None:
            return nn.Linear(in_dim, out_dim)
        chunk = self.layer_chunk
        return nn.Sequential(
            nn.Linear(in_dim, chunk),
            nn.Linear(chunk, out_dim),
        )

    def _build_bias(self, in_dim: int, out_dim: int) -> nn.Module:
        if not self.is_xctx or self.layer_chunk is None:
            return nn.Linear(in_dim, out_dim)
        chunk = self.layer_chunk
        return nn.Sequential(
            nn.Linear(in_dim, chunk),
            nn.Linear(chunk, out_dim),
        )

    def project(self, context: torch.Tensor) -> List[torch.Tensor]:
        if self.disabled:
            raise RuntimeError("Context channel disabled; project should not be called.")
        if self.disabled:
            raise RuntimeError("Context channel disabled; project should not be called.")
        return [gen(context) for gen in self.context_bias_gen]

    def update(
        self,
        block_inputs: List[torch.Tensor],
        *,
        prev_context: torch.Tensor | None,
        stop_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.disabled:
            raise RuntimeError("Context channel disabled; update should not be called.")
        pieces = [inp.detach() if stop_grad else inp for inp in block_inputs]
        messages = []
        for ln, sampler, part in zip(self.pre_norms, self.context_sampler, pieces):
            messages.append(sampler(ln(part)))
        fused = torch.stack(messages, dim=0).sum(dim=0)
        if prev_context is not None:
            detach_prev = stop_grad and self.detach_context
            residual = prev_context.detach() if detach_prev else prev_context
            fused = fused + residual
        normed_fused = self.context_fuse_norm(fused)
        context = self.context_mlp(normed_fused)
        if prev_context is not None:
            context = context + residual
        raw_context = context
        context = self.context_norm(context)
        return context, raw_context


class GRCEGPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.core = GPTCore(config)
        self.context_channels = nn.ModuleList()
        if config.n_grce > 0:
            self.context_channels.append(
                GRCEContextChannel(config, width=config.n_grce, is_xctx=False)
            )
        if config.n_xctx > 0:
            self.context_channels.append(
                GRCEContextChannel(config, width=config.n_xctx, is_xctx=True)
            )

    def forward_autoreg(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        disable_context: bool = False,
        capture_activations: bool = False,
        think_token_id: int | None = None,
        collect_relu_mask: bool = False,
        context_disabled_rows: torch.Tensor | None = None,
        xctx_disabled_rows: torch.Tensor | None = None,
        context_dropout_positions: torch.Tensor | None = None,
        disable_xctx: bool = False,
        attention_disabled_rows: torch.Tensor | None = None,
        attention_dropout_positions: torch.Tensor | None = None,
        initial_context_raw: list[torch.Tensor] | None = None,
        record_final_context: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        B, T = idx.shape
        device = idx.device
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
        if xctx_disabled_rows is not None:
            xctx_disabled_rows = xctx_disabled_rows.to(device=device, dtype=torch.bool)
        if xctx_disabled_rows is None or not use_context:
            xctx_disabled_rows = torch.zeros(B, dtype=torch.bool, device=device)
        if context_dropout_positions is not None:
            context_dropout_positions = context_dropout_positions.to(device=device, dtype=torch.long).clone()
        if attention_disabled_rows is not None:
            attention_disabled_rows = attention_disabled_rows.to(device=device, dtype=torch.bool)
        else:
            attention_disabled_rows = torch.zeros(B, dtype=torch.bool, device=device)
        if attention_dropout_positions is not None:
            attention_dropout_positions = attention_dropout_positions.to(device=device, dtype=torch.long).clone()
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
                    context_states[idx_ch] = channel.context_norm(init_raw)
        logits_steps = []
        hidden_steps = []
        pos_counters = torch.zeros(B, dtype=torch.long, device=device)
        last_content_token = torch.full(
            (B,),
            -1,
            dtype=torch.long,
            device=device,
        )
        think_marker = None
        if think_token_id is not None:
            think_marker = self.core.tok_emb.weight[think_token_id]
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

        for t in range(T):
            xctx_step_mask = None
            xctx_mask_has = False
            if context_dropout_positions is not None:
                step_mask = context_dropout_positions == t
                if step_mask.any():
                    context_dropout_positions[step_mask] = -1
                    xctx_step_mask = step_mask
                    xctx_mask_has = True
            prefix = idx[:, : t + 1]
            token_ids = prefix[:, -1]
            tok_last = self.core.tok_emb(token_ids)
            think_mask = None
            if think_token_id is not None:
                think_mask = token_ids == think_token_id
                if think_mask.any():
                    tok_last = tok_last.clone()
                    valid_prev = last_content_token >= 0
                    if valid_prev.any():
                        prev_emb = self.core.tok_emb(last_content_token.clamp(min=0))
                        combined_mask = think_mask & valid_prev
                        if combined_mask.any():
                            tok_last[combined_mask] = prev_emb[combined_mask]
                    if think_marker is not None:
                        tok_last[think_mask] = tok_last[think_mask] + think_marker
            pos_ids = pos_counters.clone()
            if think_mask is not None and think_mask.any():
                prior_pos = torch.clamp(pos_counters[think_mask] - 1, min=0)
                pos_ids[think_mask] = prior_pos
            pos_emb = self.core.pos_emb(pos_ids)
            if think_mask is not None:
                content_mask = ~think_mask
            else:
                content_mask = torch.ones_like(token_ids, dtype=torch.bool)
            if content_mask.any():
                last_content_token[content_mask] = token_ids[content_mask]
            pos_counters = pos_counters + content_mask.to(pos_counters.dtype)
            token_input = tok_last + pos_emb
            block_biases = None
            if use_context:
                for channel, state in zip(active_channels, context_states):
                    chunk = channel.layer_chunk if channel.is_xctx else None
                    channel_mask = base_context_mask
                    channel_mask_has = base_context_mask_has
                    xctx_len = None
                    if channel.is_xctx:
                        if base_xctx_mask_has:
                            channel_mask = (
                                (channel_mask | base_xctx_mask)
                                if channel_mask_has
                                else base_xctx_mask
                            )
                            channel_mask_has = True
                        if chunk is not None:
                            xctx_len = chunk
                            if xctx_mask_has:
                                channel_mask = (
                                    (channel_mask | xctx_step_mask)
                                    if channel_mask_has
                                    else xctx_step_mask
                                )
                                channel_mask_has = True
                    state_for_bias = state
                    if channel_mask_has:
                        state_for_bias = state_for_bias.clone()
                        state_for_bias[channel_mask] = 0
                    bias_vectors = channel.project(state_for_bias)
                    channel_biases: list[torch.Tensor] = []
                    for bias_vec in bias_vectors:
                        if channel_mask_has:
                            bias_vec = bias_vec.clone()
                            bias_vec[channel_mask] = 0
                        full = torch.zeros(
                            B,
                            prefix.size(1),
                            self.config.n_embd,
                            device=device,
                            dtype=bias_vec.dtype,
                        )
                        full[:, -1, :] = bias_vec
                        channel_biases.append(full)
                    if block_biases is None:
                        block_biases = channel_biases
                    else:
                        for layer_idx in range(len(block_biases)):
                            block_biases[layer_idx] = (
                                block_biases[layer_idx] + channel_biases[layer_idx]
                            )
            logits, hidden_layer, block_inputs, layer_masks = self.core(
                prefix,
                block_biases=block_biases,
                record_relu_mask=collect_relu_mask,
                attention_disabled_rows=attention_disabled_rows,
                attention_dropout_positions=attention_dropout_positions,
            )
            if relu_activity is not None:
                relu_activity.append(layer_masks)
            if activation_store is not None:
                for layer_idx, block_inp in enumerate(block_inputs):
                    norms = torch.linalg.vector_norm(block_inp.detach(), dim=-1)
                    activation_store["block_norms"][layer_idx].extend(
                        norms.cpu().tolist()
                    )
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
                    chunk = channel.layer_chunk if channel.is_xctx else None
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
            logits_steps.append(logits[:, -1:, :])
            hidden_steps.append(hidden_layer[:, -1:, :])
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


def build_loss_targets(
    targets: torch.Tensor,
    think: ThinkSettings | None,
    random_mask: torch.Tensor | None,
) -> torch.Tensor:
    mask: torch.Tensor | None = None
    if think is not None and think.enabled and think.token_id is not None:
        mask = targets == think.token_id
    if random_mask is not None:
        mask = random_mask if mask is None else (mask | random_mask)
    if mask is None or not mask.any():
        return targets
    masked = targets.clone()
    masked[mask] = LOSS_IGNORE_INDEX
    return masked


def expand_prompt_with_thinking(
    model: GRCEGPT, prompt: torch.Tensor, think: ThinkSettings | None
) -> torch.Tensor:
    if think is None or not think.enabled or think.token_id is None:
        return prompt
    if prompt.size(0) != 1:
        return prompt
    idx = prompt.clone()
    inserted = 0
    pos = 0
    max_insertions = max(0, think.max_steps)
    max_positions = int(model.config.block_size)
    while (
        pos < idx.size(1)
        and inserted < max_insertions
        and idx.size(1) < max_positions
    ):
        prefix = idx[:, : pos + 1]
        logits, _, _ = model.forward_autoreg(
            prefix,
            think_token_id=active_think_token_id(think),
        )
        next_logits = logits[:, -1, :]
        top_ids = torch.argmax(next_logits, dim=-1)
        if int(top_ids.item()) == int(think.token_id):
            think_tok = torch.tensor([[think.token_id]], dtype=idx.dtype, device=idx.device)
            idx = torch.cat((idx[:, : pos + 1], think_tok, idx[:, pos + 1 :]), dim=1)
            inserted += 1
            pos += 1
            if idx.size(1) >= max_positions:
                break
            continue
        pos += 1
    return idx


# -----------------------------------------------------------------------------
# Training / Generation Helpers
# -----------------------------------------------------------------------------


def evaluate_split(
    model: GRCEGPT,
    dataset: TextDataset,
    device: torch.device,
    block_size: int,
    batch_size: int,
    split: str,
    iters: int,
    *,
    disable_context: bool = False,
    disable_xctx: bool = False,
    disable_attention: bool = False,
    think_settings: ThinkSettings | None = None,
    undo_settings: UndoSettings | None = None,
    batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    use_special_rows: bool = False,
) -> float:
    ce_losses = []
    if batches is None:
        batches = []
        for _ in range(iters):
            xb, yb = dataset.get_batch(
                split,
                block_size,
                batch_size,
                device,
            )
            batches.append((xb, yb))
    think_token_id = active_think_token_id(think_settings)
    context_path_enabled = bool(model.context_channels)
    xctx_enabled = any(
        getattr(channel, "is_xctx", False) for channel in getattr(model, "context_channels", [])
    )
    think_enabled = think_settings is not None and think_settings.enabled
    for payload in batches:
        xb, yb = payload[:2]
        init_ctx = None
        special_masks: SpecialRowMasks | None = None
        if use_special_rows:
            special_masks = build_special_row_masks(
                xb.size(0),
                block_size,
                xb.device,
                think_enabled=think_enabled,
                context_enabled=context_path_enabled,
                xctx_enabled=xctx_enabled,
            )
        context_special_rows = (
            special_masks.context_special_rows if special_masks is not None else None
        )
        think_disabled_rows = (
            set(special_masks.think_disabled_rows) if special_masks is not None else None
        )
        forced_think_rows = (
            set(special_masks.forced_think_rows) if special_masks is not None else None
        )
        aug_xb, aug_yb, random_mask, _, think_slot_mask = augment_training_batch(
            model,
            xb,
            yb,
            think_settings,
            undo_settings,
            disable_context_rows=context_special_rows,
            disable_think_rows=think_disabled_rows,
            forced_think_rows=forced_think_rows,
            initial_context_raw=init_ctx,
        )
        attention_disabled_rows = None
        if disable_attention:
            attention_disabled_rows = torch.ones(
                aug_xb.size(0), dtype=torch.bool, device=device
            )
        elif special_masks is not None:
            attention_disabled_rows = special_masks.attention_disabled_mask
        attention_dropout_positions = None
        if disable_attention:
            attention_dropout_positions = attention_disabled_rows
        elif special_masks is not None:
            attention_dropout_positions = special_masks.attention_dropout_positions
        context_disabled_rows = None
        xctx_disabled_rows = None
        context_dropout_positions = None
        if special_masks is not None:
            context_disabled_rows = special_masks.context_disabled_mask
            xctx_disabled_rows = special_masks.xctx_disabled_mask
            context_dropout_positions = special_masks.context_dropout_positions
        raw_logits, _, _ = model.forward_autoreg(
            aug_xb,
            disable_context=disable_context,
            think_token_id=think_token_id,
            disable_xctx=disable_xctx,
            attention_disabled_rows=attention_disabled_rows,
            attention_dropout_positions=attention_dropout_positions,
            context_disabled_rows=context_disabled_rows,
            xctx_disabled_rows=xctx_disabled_rows,
            context_dropout_positions=context_dropout_positions,
            initial_context_raw=init_ctx,
        )
        logits_main = disable_think_logits(raw_logits.clone(), think_settings)
        logits_main = apply_think_slot_mask(logits_main, think_slot_mask, think_settings)
        loss_targets = build_loss_targets(aug_yb, think_settings, random_mask)
        logits_flat = logits_main.view(-1, logits_main.size(-1))
        per_token = F.cross_entropy(
            logits_flat,
            loss_targets.view(-1),
            reduction="none",
            ignore_index=LOSS_IGNORE_INDEX,
        )
        valid_mask = loss_targets != LOSS_IGNORE_INDEX
        denom = valid_mask.sum().item()
        if denom == 0:
            main_loss = per_token.sum() * 0
        else:
            main_loss = per_token.sum() / denom
        ce_losses.append(main_loss.item())
    ce_avg = sum(ce_losses) / len(ce_losses)
    return ce_avg


def build_think_sequences(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    counts: torch.Tensor,
    *,
    think_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T = inputs.shape
    device = inputs.device
    dtype = inputs.dtype
    new_inputs = torch.empty_like(inputs)
    new_targets = torch.empty_like(targets)
    eval_targets = torch.full_like(targets, LOSS_IGNORE_INDEX)
    eval_weights = torch.zeros(B, T, device=device, dtype=torch.float32)

    limit = T + 1
    counts = counts.to(device=device, dtype=torch.long).clamp_min(0)

    for row in range(B):
        seq_entries: list[dict] = []
        base_targets = [int(val) for val in targets[row].tolist()]
        for idx in range(T):
            seq_entries.append(
                {"token": int(inputs[row, idx].item()), "base_index": idx, "tag": "base"}
            )
        tail_token = int(targets[row, -1].item())
        seq_entries.append({"token": tail_token, "base_index": None, "tag": "tail"})

        def truncate() -> None:
            if len(seq_entries) > limit:
                del seq_entries[limit:]

        base_counts = counts[row]
        for base_idx in range(T):
            count = int(base_counts[base_idx].item())
            if count <= 0:
                continue
            insert_pos = None
            for pos, entry in enumerate(seq_entries):
                if entry.get("base_index") == base_idx:
                    insert_pos = pos + 1
                    break
            if insert_pos is None:
                continue
            for k in range(count):
                seq_entries.insert(
                    insert_pos + k,
                    {
                        "token": int(think_token_id),
                        "tag": "think",
                        "base_index": base_idx,
                        "think_idx": k,
                        "think_total": count,
                    },
                )
                truncate()

        while len(seq_entries) < limit:
            seq_entries.append(seq_entries[-1])

        row_tokens = torch.tensor(
            [entry["token"] for entry in seq_entries],
            dtype=dtype,
            device=device,
        )
        new_inputs[row] = row_tokens[:-1]
        new_targets[row] = row_tokens[1:]

        for pos, entry in enumerate(seq_entries[:-1]):
            if entry.get("tag") != "base":
                continue
            base_idx = entry.get("base_index")
            if base_idx is None or base_idx >= len(base_targets):
                continue
            count = int(base_counts[base_idx].item())
            eval_targets[row, pos] = base_targets[base_idx]
            eval_weights[row, pos] = float(max(1, count + 1))

    return new_inputs, new_targets, eval_targets, eval_weights


def run_think_insertion_eval(
    model: GRCEGPT,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    counts: torch.Tensor,
    *,
    think_token_id: int,
    initial_context_raw: list[torch.Tensor] | None = None,
) -> tuple[float, float]:
    new_inputs, new_targets, eval_targets, eval_weights = build_think_sequences(
        inputs,
        targets,
        counts,
        think_token_id=think_token_id,
    )
    logits, _, _ = model.forward_autoreg(
        new_inputs,
        think_token_id=think_token_id,
        initial_context_raw=initial_context_raw,
    )
    think_guard = ThinkSettings(max_steps=1, token_id=think_token_id)
    logits = disable_think_logits(logits, think_guard)
    masked_targets = torch.where(
        eval_weights > 0,
        eval_targets,
        torch.full_like(eval_targets, LOSS_IGNORE_INDEX),
    )
    loss_flat = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        masked_targets.view(-1),
        reduction="none",
        ignore_index=LOSS_IGNORE_INDEX,
    )
    weighted = loss_flat * eval_weights.view(-1)
    total_weight = float(eval_weights.sum().item())
    if total_weight <= 0:
        return 0.0, 0.0
    return float(weighted.sum().item()), total_weight


def evaluate_think_modes(
    model: GRCEGPT,
    batches: list[
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]
    ],
    *,
    think_token_id: int,
) -> dict[str, float]:
    results: dict[str, float] = {}

    def aggregate(batch_losses: list[tuple[float, float]]) -> float:
        weight_sum = sum(weight for _, weight in batch_losses)
        if weight_sum <= 0:
            return 0.0
        loss_sum = sum(loss for loss, _ in batch_losses)
        return loss_sum / weight_sum

    batch_losses: list[tuple[float, float]] = []
    for payload in batches:
        if len(payload) == 3:
            xb, yb, init_ctx = payload
        else:
            xb, yb = payload[:2]
            init_ctx = None
        logits, _, _ = model.forward_autoreg(
            xb,
            think_token_id=think_token_id,
            initial_context_raw=init_ctx,
        )
        preds = torch.argmax(logits, dim=-1)
        counts = preds.eq(think_token_id).long()
        counts = counts * (yb != LOSS_IGNORE_INDEX).long()
        loss_sum, weight = run_think_insertion_eval(
            model,
            xb,
            yb,
            counts,
            think_token_id=think_token_id,
            initial_context_raw=init_ctx,
        )
        if weight > 0:
            batch_losses.append((loss_sum, weight))
    results["think"] = aggregate(batch_losses)

    const_batches: list[tuple[float, float]] = []
    for payload in batches:
        if len(payload) == 3:
            xb, yb, init_ctx = payload
        else:
            xb, yb = payload[:2]
            init_ctx = None
        counts = torch.ones_like(xb, dtype=torch.long)
        counts = counts * (yb != LOSS_IGNORE_INDEX).long()
        loss_sum, weight = run_think_insertion_eval(
            model,
            xb,
            yb,
            counts,
            think_token_id=think_token_id,
            initial_context_raw=init_ctx,
        )
        if weight > 0:
            const_batches.append((loss_sum, weight))
    results["think2x"] = aggregate(const_batches)

    triple_batches: list[tuple[float, float]] = []
    for payload in batches:
        if len(payload) == 3:
            xb, yb, init_ctx = payload
        else:
            xb, yb = payload[:2]
            init_ctx = None
        counts = torch.full_like(xb, 2, dtype=torch.long)
        counts = counts * (yb != LOSS_IGNORE_INDEX).long()
        loss_sum, weight = run_think_insertion_eval(
            model,
            xb,
            yb,
            counts,
            think_token_id=think_token_id,
            initial_context_raw=init_ctx,
        )
        if weight > 0:
            triple_batches.append((loss_sum, weight))
    results["think3x"] = aggregate(triple_batches)

    return results


def train_model(
    model: GRCEGPT,
    dataset: TextDataset,
    device: torch.device,
    steps: int,
    block_size: int,
    batch_size: int,
    eval_interval: int,
    eval_iters: int,
    start_step: int,
    sample_prompt: torch.Tensor,
    sample_chars: int,
    tokenizer: GPT2TokenizerWrapper,
    suppress_newlines: bool,
    newline_token_id: int | None,
    think_settings: ThinkSettings | None,
    suppress_think_output: bool,
    suppress_think_prompt: bool,
    think_hard: bool,
    undo_settings: UndoSettings | None,
    prompt_tracker: PromptTracker | None = None,
    reset_prompt_queue: bool = False,
    *,
    reward_relu: bool = False,
    context_dropout_interval: int = 1,
    cycle_wall_start: float,
    base_wall_seconds: float,
    show_time: bool = False,
    underline_tokens: bool = False,
    default_prompt_boundary: bool = False,
    boundary_blocklist: Sequence[int] | None = None,
    show_train_loss_details: bool = False,
    show_test_loss_details: bool = True,
    long_loss_log: bool = False,
    full_eval_stride: int = 1,
) -> Tuple[int, List[Dict[str, float]], float, float, float, float]:
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    total_steps = start_step
    history_updates: List[Dict[str, float]] = []
    printed_header = False
    long_log_force = bool(long_loss_log)
    think_enabled = think_settings is not None and think_settings.enabled
    undo_enabled = undo_settings is not None and undo_settings.enabled
    think_token_id = active_think_token_id(think_settings)
    noun_detector = (
        NounExpectationDetector(model, tokenizer, think_token_id)
        if underline_tokens
        else None
    )
    reward_tracker = (
        ReLURewardTracker(model, steps, scale=reward_relu)
        if reward_relu > 0
        else None
    )
    if prompt_tracker is not None:
        prompt_queue = prompt_tracker.prompt_queue(reset=reset_prompt_queue)
    else:
        prompt_queue = []

    def enqueue_prompt(idx: int, *, front: bool = False) -> None:
        if prompt_tracker is None or idx is None or idx < 0:
            return
        prompt_queue[:] = [existing for existing in prompt_queue if existing != idx]
        if front:
            prompt_queue.insert(0, idx)
        else:
            prompt_queue.append(idx)

    full_eval_stride = max(0, int(full_eval_stride))
    full_eval_enabled = full_eval_stride > 0
    show_think_columns = bool(
        full_eval_enabled and think_enabled and tokenizer.think_id is not None
    )
    show_train_details = bool(show_train_loss_details and full_eval_enabled)
    show_test_details = bool(show_test_loss_details and full_eval_enabled)
    context_dropout_interval = max(0, int(context_dropout_interval))
    context_path_enabled = bool(model.context_channels)
    xctx_enabled = any(
        getattr(channel, "is_xctx", False) for channel in getattr(model, "context_channels", [])
    )
    loop_wall_start = time.time()
    loop_cpu_start = time.process_time()
    eval_wall_total = 0.0
    eval_cpu_total = 0.0
    preeval_wall_total = 0.0
    preeval_cpu_total = 0.0
    for step in range(1, steps + 1):
        batch_payload = dataset.get_batch(
            "train",
            block_size,
            batch_size,
            device,
        )
        xb, yb = batch_payload
        current_step_index = total_steps
        context_dropout_active = (
            context_path_enabled
            and context_dropout_interval > 0
            and current_step_index % context_dropout_interval == 0
        )
        context_special_rows: set[int] = set()
        context_disabled_mask: torch.Tensor | None = None
        xctx_disabled_mask: torch.Tensor | None = None
        context_dropout_positions: torch.Tensor | None = None
        attention_disabled_mask: torch.Tensor | None = None
        attention_dropout_positions: torch.Tensor | None = None
        think_disabled_rows: set[int] = set()
        forced_think_rows: set[int] = set()
        if context_dropout_active:
            special_masks = build_special_row_masks(
                batch_size,
                block_size,
                device,
                think_enabled=think_enabled,
                context_enabled=context_path_enabled,
                xctx_enabled=xctx_enabled,
            )
            context_special_rows = set(special_masks.context_special_rows)
            context_disabled_mask = special_masks.context_disabled_mask
            xctx_disabled_mask = special_masks.xctx_disabled_mask
            context_dropout_positions = special_masks.context_dropout_positions
            attention_disabled_mask = special_masks.attention_disabled_mask
            attention_dropout_positions = special_masks.attention_dropout_positions
            think_disabled_rows = set(special_masks.think_disabled_rows)
            forced_think_rows = set(special_masks.forced_think_rows)
        elif think_enabled and batch_size > 0:
            think_disabled_rows.add(random.randrange(batch_size))
        xb, yb, random_mask, think_labels, think_slot_mask = augment_training_batch(
            model,
            xb,
            yb,
            think_settings,
            undo_settings,
            disable_context_rows=context_special_rows,
            disable_think_rows=think_disabled_rows,
            forced_think_rows=forced_think_rows,
            initial_context_raw=None,
        )
        logits, hidden_states, activation_store = model.forward_autoreg(
            xb,
            yb,
            think_token_id=think_token_id,
            collect_relu_mask=reward_tracker is not None,
            context_disabled_rows=context_disabled_mask,
            xctx_disabled_rows=xctx_disabled_mask,
            context_dropout_positions=context_dropout_positions,
            attention_disabled_rows=attention_disabled_mask,
            attention_dropout_positions=attention_dropout_positions,
            initial_context_raw=None,
        )
        raw_logits = logits
        logits_main = disable_think_logits(raw_logits.clone(), think_settings)
        logits_main = apply_think_slot_mask(logits_main, think_slot_mask, think_settings)
        loss_targets = build_loss_targets(yb, think_settings, random_mask)
        logits_flat = logits_main.view(-1, logits_main.size(-1))
        token_loss_flat = F.cross_entropy(
            logits_flat,
            loss_targets.view(-1),
            reduction="none",
            ignore_index=LOSS_IGNORE_INDEX,
        )
        token_losses = token_loss_flat.view_as(loss_targets)
        valid_mask = loss_targets != LOSS_IGNORE_INDEX
        denom = valid_mask.sum().item()
        if denom == 0:
            main_loss = token_loss_flat.sum() * 0
        else:
            main_loss = token_loss_flat.sum() / denom
        align_loss = compute_think_alignment_loss(
            model,
            raw_logits,
            xb,
            loss_targets,
            token_losses.detach(),
            think_settings,
        )
        loss = main_loss + align_loss
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_steps += 1

        if reward_tracker is not None:
            relu_activity = None
            if activation_store is not None:
                relu_activity = activation_store.get("relu_activity")
            if relu_activity is not None:
                with torch.no_grad():
                    reward_tracker.record_batch(
                        relu_activity,
                        token_losses.detach(),
                        valid_mask.detach(),
                    )
                    reward_tracker.maybe_apply()

        preeval_wall_block = time.time()
        preeval_cpu_block = time.process_time()
        if step == 1 or step % eval_interval == 0 or step == steps:
            full_eval_now = full_eval_enabled and (
                full_eval_stride > 0 and total_steps % full_eval_stride == 0
            )
            eval_wall_block = time.time()
            eval_cpu_block = time.process_time()
            model.eval()
            with torch.no_grad():
                split_metrics: dict[str, dict[str, float]] = {"train": {}, "test": {}}
                cached_batches: dict[
                    str, list[tuple[torch.Tensor, torch.Tensor]]
                ] = {}
                for split in ("train", "test"):
                    cached_batches[split] = []
                    for _ in range(eval_iters):
                        cached_batch = dataset.get_batch(
                            split,
                            block_size,
                            batch_size,
                            device,
                        )
                        bx, by = cached_batch
                        cached_batches[split].append((bx, by))
                    base_batches = cached_batches[split]
                    split_metrics[split]["with_think"] = float(
                        evaluate_split(
                            model,
                            dataset,
                            device,
                            block_size,
                            batch_size,
                            split,
                            eval_iters,
                            disable_context=False,
                            disable_xctx=False,
                            disable_attention=False,
                            think_settings=think_settings,
                            undo_settings=undo_settings,
                            batches=base_batches,
                        )
                    )
                    if not full_eval_now:
                        continue
                    split_metrics[split]["with_think_special"] = float(
                        evaluate_split(
                            model,
                            dataset,
                            device,
                            block_size,
                            batch_size,
                            split,
                            eval_iters,
                            disable_context=False,
                            disable_xctx=False,
                            disable_attention=False,
                            think_settings=think_settings,
                            undo_settings=undo_settings,
                            batches=base_batches,
                            use_special_rows=True,
                        )
                    )
                    noprev_batches = list(base_batches)
                    split_metrics[split]["with_think_noprev"] = float(
                        evaluate_split(
                            model,
                            dataset,
                            device,
                            block_size,
                            batch_size,
                            split,
                            eval_iters,
                            disable_context=False,
                            disable_xctx=False,
                            disable_attention=False,
                            think_settings=think_settings,
                            undo_settings=undo_settings,
                            batches=noprev_batches,
                        )
                    )
                    plain_kwargs = dict(
                        think_settings=None,
                        undo_settings=None,
                        batches=cached_batches[split],
                    )
                    split_metrics[split]["plain"] = float(
                        evaluate_split(
                            model,
                            dataset,
                            device,
                            block_size,
                            batch_size,
                            split,
                            eval_iters,
                            disable_context=False,
                            disable_xctx=False,
                            disable_attention=False,
                            **plain_kwargs,
                        )
                    )
                    split_metrics[split]["plain_noctx"] = float(
                        evaluate_split(
                            model,
                            dataset,
                            device,
                            block_size,
                            batch_size,
                            split,
                            eval_iters,
                            disable_context=False,
                            disable_xctx=True,
                            disable_attention=False,
                            **plain_kwargs,
                        )
                    )
                    split_metrics[split]["plain_noatt"] = float(
                        evaluate_split(
                            model,
                            dataset,
                            device,
                            block_size,
                            batch_size,
                            split,
                            eval_iters,
                            disable_context=False,
                            disable_xctx=False,
                            disable_attention=True,
                            **plain_kwargs,
                        )
                    )
                    split_metrics[split]["plain_none"] = float(
                        evaluate_split(
                            model,
                            dataset,
                            device,
                            block_size,
                            batch_size,
                            split,
                            eval_iters,
                            disable_context=False,
                            disable_xctx=True,
                            disable_attention=True,
                            **plain_kwargs,
                        )
                    )
                    if show_think_columns and full_eval_now:
                        think_modes = evaluate_think_modes(
                            model,
                            cached_batches[split],
                            think_token_id=int(tokenizer.think_id),
                        )
                        split_metrics[split].update(think_modes)
            block_wall = time.time() - eval_wall_block
            block_cpu = time.process_time() - eval_cpu_block
            eval_wall_total += block_wall
            eval_cpu_total += block_cpu
            preeval_wall_total += time.time() - preeval_wall_block
            preeval_cpu_total += time.process_time() - preeval_cpu_block
            prompt_input = sample_prompt
            prompt_needs_boundary_flag = (
                default_prompt_boundary and boundary_blocklist is not None
            )
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
                        boundary_blocklist is not None
                        and prompt_needs_boundary(prompt_text)
                    )
                    state_val = prompt_tracker.state(current_prompt_idx)
                    if state_val == 1:
                        use_argmax_completion = True
                    else:
                        use_argmax_completion = random.random() < 0.5
                    sampling_strategy = (
                        "argmax" if use_argmax_completion else "sample"
                    )
                else:
                    use_argmax_completion = random.random() < 0.5
                    sampling_strategy = (
                        "argmax" if use_argmax_completion else "sample"
                    )
            sample_tokens, prompt_len = generate(
                model,
                prompt_input.clone(),
                sample_chars,
                suppress_newlines=suppress_newlines,
                newline_token_id=newline_token_id,
                think_settings=think_settings,
                suppress_think=suppress_think_output,
                suppress_think_prompt=suppress_think_prompt,
                think_hard=think_hard,
                first_token_blocklist=(
                    boundary_blocklist if prompt_needs_boundary_flag else None
                ),
                sampling_strategy=sampling_strategy,
            )
            model.train()
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
                        print(
                            color_text(
                                f"Prompt #{current_prompt_idx + 1} satisfied ({mode}): {prompt_text} (expected '{expected}')",
                                Colors.YELLOW,
                                bold=True,
                            )
                        )
                    else:
                        mode = "sample"
                        print(
                            color_text(
                                f"Prompt #{current_prompt_idx + 1} satisfied ({mode}): {prompt_text} (expected '{expected}')",
                                Colors.YELLOW,
                                bold=False,
                            )
                        )
                if new_state < 2:
                    front_requeue = new_state == 1 and prev_state == 0
                    enqueue_prompt(
                        current_prompt_idx,
                        front=front_requeue,
                    )

            prefix_text = color_tokens(
                tokenizer,
                prompt_ids,
                [Colors.MAGENTA, Colors.GREEN],
                bold=False,
                think_token_id=tokenizer.think_id,
                undo_token_id=tokenizer.undo_id,
                noun_detector=noun_detector,
            )
            completion_text = color_tokens(
                tokenizer,
                completion_ids,
                [Colors.YELLOW, Colors.CYAN],
                bold=use_argmax_completion,
                think_token_id=tokenizer.think_id,
                undo_token_id=tokenizer.undo_id,
                noun_detector=noun_detector,
                context_prefix=prompt_ids,
            )
            colored_sample = prefix_text + completion_text
            long_log_now = bool(long_log_force or full_eval_now)
            if not printed_header:
                if prompt_tracker is not None:
                    random_only_count, solved_count, total_prompts = prompt_tracker.counts()
                else:
                    total_prompts = len(default_prompt_entries())
                    random_only_count = 0
                    solved_count = 0
                train_header = "train loss"
                if show_train_details:
                    train_header += " special noprev : normal noctx noatt none"
                    if show_think_columns:
                        train_header += " : think 2x 3x"
                test_header = "test loss"
                if show_test_details:
                    test_header += " special noprev : normal noctx noatt none"
                    if show_think_columns:
                        test_header += " : think 2x 3x"
                header_parts: List[str] = []
                if show_time:
                    header_parts.append("time")
                header_parts.append(color_text("step", Colors.CYAN))
                header_parts.append(color_text(train_header, Colors.MAGENTA))
                header_parts.append(color_text(test_header, Colors.GREEN))
                header_line = " | ".join(header_parts) + color_text(
                    f" | sample ({random_only_count}/{solved_count}/{total_prompts})",
                    Colors.YELLOW,
                )
                print(header_line)
                printed_header = True

            def format_metric(split: str, key: str) -> str:
                value = split_metrics.get(split, {}).get(key)
                if value is None or math.isnan(value):
                    return " -- "
                return f"{value:.2f}"

            def format_train_line() -> str:
                if not show_train_details or not long_log_now:
                    return format_metric("train", "with_think")
                primary_group = " ".join(
                    format_metric("train", key)
                    for key in (
                        "with_think",
                        "with_think_special",
                        "with_think_noprev",
                    )
                )
                diag_vals = " ".join(
                    format_metric("train", key)
                    for key in ("plain", "plain_noctx", "plain_noatt", "plain_none")
                )
                parts = [primary_group, diag_vals]
                if show_think_columns and long_log_now:
                    think_vals = " ".join(
                        format_metric("train", key)
                        for key in ("think", "think2x", "think3x")
                    )
                    parts.append(think_vals)
                return " : ".join(parts)

            def format_test_line() -> str:
                if not show_test_details or not long_log_now:
                    return format_metric("test", "with_think")
                primary_group = " ".join(
                    format_metric("test", key)
                    for key in (
                        "with_think",
                        "with_think_special",
                        "with_think_noprev",
                    )
                )
                diag_vals = " ".join(
                    format_metric("test", key)
                    for key in ("plain", "plain_noctx", "plain_noatt", "plain_none")
                )
                parts = [primary_group, diag_vals]
                if show_think_columns and long_log_now:
                    think_vals = " ".join(
                        format_metric("test", key)
                        for key in ("think", "think2x", "think3x")
                    )
                    parts.append(think_vals)
                return " : ".join(parts)

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
            line = " | ".join(line_parts) + " | " + colored_sample
            print(line)
            eval_now = time.time()
            cycle_wall_elapsed = max(0.0, eval_now - cycle_wall_start)
            total_wall_seconds = base_wall_seconds + cycle_wall_elapsed

            record = {
                "step": total_steps,
                "train_loss": float(split_metrics["train"].get("with_think", 0.0)),
                "test_loss": float(split_metrics["test"].get("with_think", 0.0)),
                "train_wall_seconds": float(total_wall_seconds),
                "unix_time": float(eval_now),
                "train_cursor": int(dataset.positions.get("train", 0)),
                "test_cursor": int(dataset.positions.get("test", 0)),
            }
            if full_eval_now:
                record.update(
                    {
                        "train_loss_special": float(
                            split_metrics["train"].get("with_think_special", 0.0)
                        ),
                        "train_loss_noprev": float(
                            split_metrics["train"].get("with_think_noprev", 0.0)
                        ),
                        "train_loss_plain": float(split_metrics["train"].get("plain", 0.0)),
                        "train_loss_noctx": float(
                            split_metrics["train"].get("plain_noctx", 0.0)
                        ),
                        "train_loss_noatt": float(
                            split_metrics["train"].get("plain_noatt", 0.0)
                        ),
                        "train_loss_none": float(
                            split_metrics["train"].get("plain_none", 0.0)
                        ),
                        "test_loss_special": float(
                            split_metrics["test"].get("with_think_special", 0.0)
                        ),
                        "test_loss_noprev": float(
                            split_metrics["test"].get("with_think_noprev", 0.0)
                        ),
                        "test_loss_plain": float(split_metrics["test"].get("plain", 0.0)),
                        "test_loss_noctx": float(
                            split_metrics["test"].get("plain_noctx", 0.0)
                        ),
                        "test_loss_noatt": float(
                            split_metrics["test"].get("plain_noatt", 0.0)
                        ),
                        "test_loss_none": float(
                            split_metrics["test"].get("plain_none", 0.0)
                        ),
                    }
                )
            if show_think_columns and full_eval_now:
                record["train_loss_think"] = float(split_metrics["train"].get("think", 0.0))
                record["train_loss_think2x"] = float(split_metrics["train"].get("think2x", 0.0))
                record["train_loss_think3x"] = float(split_metrics["train"].get("think3x", 0.0))
                record["test_loss_think"] = float(split_metrics["test"].get("think", 0.0))
                record["test_loss_think2x"] = float(split_metrics["test"].get("think2x", 0.0))
                record["test_loss_think3x"] = float(split_metrics["test"].get("think3x", 0.0))
            history_updates.append(record)
    
    if reward_tracker is not None:
        reward_tracker.finalize()
    loop_wall_total = time.time() - loop_wall_start
    loop_cpu_total = time.process_time() - loop_cpu_start
    return (
        total_steps,
        history_updates,
        loop_wall_total,
        loop_cpu_total,
        eval_wall_total,
        eval_cpu_total,
        preeval_wall_total,
        preeval_cpu_total,
    )


def run_report_mode(
    model: GRCEGPT,
    tokenizer: GPT2TokenizerWrapper,
    prompt_tokens: torch.Tensor,
    sample_len: int,
    count: int,
    device: torch.device,
    suppress_newlines: bool,
    newline_token_id: int | None,
    think_settings: ThinkSettings | None,
    suppress_think: bool,
    suppress_think_prompt: bool,
    think_hard: bool,
    underline_tokens: bool = False,
    default_prompt_boundary: bool = False,
    boundary_blocklist: Sequence[int] | None = None,
) -> None:
    model.eval()
    think_token_id = active_think_token_id(think_settings)
    noun_detector = None
    if underline_tokens:
        noun_detector = NounExpectationDetector(model, tokenizer, think_token_id)
    base_len = prompt_tokens.size(1)
    with torch.no_grad():
        for idx in range(1, count + 1):
            use_argmax_completion = idx == 1
            sampling_strategy = "argmax" if use_argmax_completion else "sample"
            generated, prompt_len = generate(
                model,
                prompt_tokens.clone(),
                sample_len,
                suppress_newlines=suppress_newlines,
                newline_token_id=newline_token_id,
                think_settings=think_settings,
                suppress_think=suppress_think,
                suppress_think_prompt=suppress_think_prompt,
                think_hard=think_hard,
                first_token_blocklist=(
                    boundary_blocklist if default_prompt_boundary else None
                ),
                sampling_strategy=sampling_strategy,
            )
            tokens = generated[0].detach().cpu().tolist()
            prompt_ids = tokens[:prompt_len]
            completion_ids = tokens[prompt_len:]
            prefix_text = color_tokens(
                tokenizer,
                prompt_ids,
                [Colors.MAGENTA, Colors.GREEN],
                bold=False,
                think_token_id=tokenizer.think_id,
                undo_token_id=tokenizer.undo_id,
                noun_detector=noun_detector,
            )
            completion_text = color_tokens(
                tokenizer,
                completion_ids,
                [Colors.YELLOW, Colors.CYAN],
                bold=use_argmax_completion,
                think_token_id=tokenizer.think_id,
                undo_token_id=tokenizer.undo_id,
                noun_detector=noun_detector,
                context_prefix=prompt_ids,
            )
            print(
                color_text(f"[report {idx:02d}]", Colors.CYAN)
                + " | sample: "
                + prefix_text
                + completion_text
            )


def run_test_slice(
    *,
    dataset: TextDataset,
    tokenizer: GPT2TokenizerWrapper,
    model: GRCEGPT,
    block_size: int,
    start_pos: int | None,
    think_settings: ThinkSettings | None,
    underline_tokens: bool = False,
) -> None:
    tokens = dataset.test_tokens
    if tokens.numel() == 0:
        print(color_text("test corpus is empty", Colors.MAGENTA))
        return
    total = int(tokens.numel())
    start = int(start_pos or 0) % total
    span = block_size if block_size > 0 else 1
    indices = [int(tokens[(start + i) % total]) for i in range(span)]
    model_device = next(model.parameters()).device
    seq = torch.tensor(indices, dtype=torch.long, device=model_device).unsqueeze(0)
    model.eval()
    think_token_id = active_think_token_id(think_settings)
    noun_detector = None
    if underline_tokens:
        noun_detector = NounExpectationDetector(model, tokenizer, think_token_id)
    prefill_tokens = None
    if block_size > 0:
        pre_slice = dataset.looped_slice("test", start - block_size, block_size)
        prefill_tokens = pre_slice.to(model_device).unsqueeze(0)
    initial_contexts = None
    if prefill_tokens is not None:
        initial_contexts = compute_prefill_contexts(
            model,
            prefill_tokens,
            think_token_id=think_token_id,
        )
    with torch.no_grad():
        logits, _, activations = model.forward_autoreg(
            seq,
            capture_activations=True,
            think_token_id=think_token_id,
            initial_context_raw=initial_contexts,
        )
    preds = logits.argmax(dim=-1).squeeze(0).tolist()
    correct_mask = [False] * len(indices)
    for i in range(1, len(indices)):
        correct_mask[i] = preds[i - 1] == indices[i]

    augmented = list(indices)
    think_enabled = think_settings is not None and think_settings.enabled
    if think_enabled:
        augmented = []
        for tok, pred in zip(indices, preds):
            augmented.append(tok)
            if pred == tokenizer.think_id:
                augmented.append(tokenizer.think_id)
    tensor = torch.tensor(augmented, dtype=torch.long)
    decoded = color_tokens(
        tokenizer,
        tensor.tolist(),
        [Colors.MAGENTA, Colors.GREEN],
        bold=False,
        think_token_id=tokenizer.think_id,
        undo_token_id=tokenizer.undo_id,
        noun_detector=noun_detector,
    )
    print(
        color_text(
            f"\nTest mode: cursor={start} span={span} (wrap @ {total})",
            Colors.BLUE,
        )
    )
    print("token_ids:", " ".join(str(idx) for idx in indices))
    baseline = color_tokens(
        tokenizer,
        indices,
        [Colors.MAGENTA, Colors.GREEN],
        bold=False,
        think_token_id=tokenizer.think_id,
        undo_token_id=tokenizer.undo_id,
        correct_mask=correct_mask,
        completion_colors=[Colors.YELLOW, Colors.CYAN],
        bold_correct=True,
        noun_detector=noun_detector,
    )
    print("decoded:", baseline)
    prob_matrix = torch.softmax(logits, dim=-1).squeeze(0)
    top_k = min(10, prob_matrix.size(-1))
    rows = []
    label_width = 0
    pred_widths = [0] * top_k
    usable = max(0, len(indices) - 1)
    for pos in range(usable):
        context_id = indices[pos]
        target_id = indices[pos + 1]
        token_label = format_token_label(tokenizer, context_id)
        label_width = max(label_width, len(token_label))
        top_probs, top_idx = prob_matrix[pos].topk(top_k)
        entry = []
        for col, (pred_id, prob) in enumerate(zip(top_idx.tolist(), top_probs.tolist())):
            pred_label = format_token_label(tokenizer, pred_id)
            marker = "*" if pred_id == target_id else " "
            prob_str = f"{prob:.3f}".split(".")[-1]
            entry.append((pred_label, prob_str, marker))
            pred_widths[col] = max(pred_widths[col], len(pred_label))
        rows.append((token_label, entry))

    for token_label, entries in rows:
        parts = []
        for (pred_label, prob_str, marker), width in zip(entries, pred_widths):
            cell = f"{pred_label:>{width}} {prob_str}{marker}"
            if marker == "*":
                cell = color_text(cell, Colors.WHITE, bold=True)
            parts.append(cell)
        print(f"{token_label:>{label_width}} | {' '.join(parts)}")
    if think_enabled:
        print("thinking:", decoded)

    def summarize(values: list[float]) -> tuple[float, float, float, float] | None:
        if not values:
            return None
        tensor = torch.tensor(values, dtype=torch.float32)
        mean = float(tensor.mean().item())
        std = float(tensor.std(unbiased=False).item())
        min_val = float(tensor.min().item())
        max_val = float(tensor.max().item())
        return mean, std, min_val, max_val

    if activations is not None:
        block_norms: list[list[float]] = activations.get("block_norms", [])
        context_norms: list[float] = activations.get("context_norms", [])
        print(color_text("\nBlock activation L2 norms (unnormalized inputs):", Colors.YELLOW))
        for layer_idx, values in enumerate(block_norms):
            stats = summarize(values)
            if stats is None:
                continue
            mean, std, min_val, max_val = stats
            print(
                f"  layer {layer_idx:02d}: mean={mean:.4f} std={std:.4f} "
                f"min={min_val:.4f} max={max_val:.4f} (n={len(values)})"
            )
        stats = summarize(context_norms)
        if stats is not None:
            mean, std, min_val, max_val = stats
            print(color_text("Context channel L2 norm (pre-LN):", Colors.YELLOW))
            print(
                f"  mean={mean:.4f} std={std:.4f} "
                f"min={min_val:.4f} max={max_val:.4f} (n={len(context_norms)})"
            )


def count_eval_calls(steps: int, eval_interval: int) -> int:
    evals = 0
    for step in range(1, steps + 1):
        if step == 1 or step == steps or (eval_interval > 0 and step % eval_interval == 0):
            evals += 1
    return max(1, evals)


def tidy(text: str, replace_newline: str = FANCY_ENTER) -> str:
    return text.replace("\n", replace_newline).replace(" ", FANCY_SPACE)

def color_tokens(
    tokenizer: GPT2TokenizerWrapper,
    tokens: list[int],
    colors: list[str],
    *,
    bold: bool = True,
    replace_newline: str = FANCY_ENTER,
    think_token_id: int | None = None,
    undo_token_id: int | None = None,
    correct_mask: list[bool] | None = None,
    completion_colors: list[str] | None = None,
    bold_correct: bool = False,
    noun_detector: NounExpectationDetector | None = None,
    context_prefix: list[int] | None = None,
) -> str:
    parts: list[str] = []
    color_index = 0
    completion_index = 0
    prefix_tokens: list[int] = list(context_prefix or [])
    in_word = False
    underline_active = False
    word_connectors = {"'", "-"}
    for idx, tok in enumerate(tokens):
        if think_token_id is not None and tok == think_token_id:
            piece = THINK_SYMBOL
            color = Colors.WHITE
            parts.append(color_text(piece, color, bold=True))
            color_index += 1
            in_word = False
            underline_active = False
            prefix_tokens.append(tok)
            continue
        if undo_token_id is not None and tok == undo_token_id:
            piece = UNDO_SYMBOL
            color = Colors.BLUE
            parts.append(color_text(piece, color, bold=True))
            color_index += 1
            in_word = False
            underline_active = False
            prefix_tokens.append(tok)
            continue
        raw_piece = tokenizer.tokenizer.decode([tok], clean_up_tokenization_spaces=False)
        if not raw_piece:
            prefix_tokens.append(tok)
            continue
        display_piece = tidy(raw_piece, replace_newline=replace_newline)
        if not display_piece:
            prefix_tokens.append(tok)
            continue
        palette = colors
        if correct_mask is not None and correct_mask[idx]:
            palette = completion_colors or [Colors.YELLOW, Colors.MAGENTA]
            color = palette[completion_index % len(palette)]
            completion_index += 1
            use_bold = True if bold_correct else bold
        else:
            color = palette[color_index % len(palette)]
            color_index += 1
            use_bold = bold
        # display_piece already computed for early checks; reuse the same string
        char_pairs = list(zip(raw_piece, display_piece))
        segments: list[tuple[str, bool]] = []
        current_segment: list[str] = []
        current_underlined = underline_active

        def flush_segment() -> None:
            nonlocal current_segment
            if not current_segment:
                return
            text = "".join(current_segment)
            segments.append((text, current_underlined))
            current_segment = []

        for raw_char, display_char in char_pairs:
            char_is_letter = raw_char.isalpha()
            char_is_word_char = char_is_letter or (in_word and raw_char in word_connectors)
            if in_word and not char_is_word_char:
                flush_segment()
                in_word = False
                underline_active = False
                current_underlined = underline_active
            if not in_word and char_is_letter:
                flush_segment()
                if noun_detector is not None:
                    underline_active = noun_detector.requires_pronoun(prefix_tokens)
                else:
                    underline_active = False
                in_word = True
                current_underlined = underline_active
            if current_underlined != underline_active:
                flush_segment()
                current_underlined = underline_active
            current_segment.append(display_char)
        flush_segment()
        if not segments:
            prefix_tokens.append(tok)
            continue
        for text, underlined in segments:
            parts.append(color_text(text, color, bold=use_bold, underline=underlined))
        prefix_tokens.append(tok)
    return "".join(parts)


def format_token_label(tokenizer: GPT2TokenizerWrapper, token_id: int, *, width: int = 10) -> str:
    piece = tokenizer.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    piece = piece.replace("\n", "\\n")
    piece = piece.replace("\t", "\\t")
    piece = piece.replace("\r", "\\r")
    if not piece.strip():
        piece = f"#{token_id}"
    if len(piece) > width:
        piece = piece[: width - 1] + "…"
    return piece


def normalize_prompt(text: str) -> str:
    return text.replace(FANCY_SPACE, " ").replace(FANCY_ENTER, "\n")


def prompt_needs_boundary(text: str) -> bool:
    trimmed = text.rstrip()
    if not trimmed:
        return False
    last = trimmed[-1].lower()
    return last in ASCII_LOWERCASE


def load_checkpoint_payload(path: pathlib.Path, device: torch.device) -> tuple[dict, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        state = upgrade_state_dict(payload["model"])
        payload["model"] = state
        meta = payload
    else:
        state = upgrade_state_dict(payload)
        meta = {}
    return state, meta


def upgrade_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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


def count_layers_from_state(state: dict[str, torch.Tensor]) -> int:
    pattern = re.compile(r"core\.blocks\.(\d+)\.")
    max_idx = -1
    for key in state.keys():
        match = pattern.search(key)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1


def build_layer_mapping(
    src_layers: int,
    dst_layers: int,
    drop_layers: list[int],
    add_layers: list[int],
    allow_trim: bool,
) -> dict[int, int | None]:
    drop_zero = {idx - 1 for idx in drop_layers}
    if any(idx < 0 or idx >= src_layers for idx in drop_zero):
        raise ValueError("--drop-layers indices must fall within the source layer range")
    if len(drop_zero) != len(drop_layers):
        raise ValueError("--drop-layers indices must be unique")
    survivors = [i for i in range(src_layers) if i not in drop_zero]

    add_zero = {idx - 1 for idx in add_layers}
    if any(idx < 0 or idx >= dst_layers for idx in add_zero):
        raise ValueError("--add-layers indices must fall within the destination layer range")
    if len(add_zero) != len(add_layers):
        raise ValueError("--add-layers indices must be unique")

    dest_non_new = dst_layers - len(add_zero)
    if dest_non_new < 0:
        raise ValueError("Too many --add-layers entries for the destination depth")

    if drop_layers or add_layers:
        if len(survivors) != dest_non_new:
            raise ValueError(
                "--drop-layers/--add-layers must leave exactly the destination layer count"
            )
    else:
        if len(survivors) < dest_non_new:
            raise ValueError(
                "Destination has more layers than source; use --add-layers to specify insertions"
            )
        if len(survivors) > dest_non_new:
            if not allow_trim:
                raise ValueError(
                    "Destination has fewer layers; rerun with --trim-model to allow trimming"
                )
            survivors = survivors[:dest_non_new]

    mapping: dict[int, int | None] = {}
    survivor_iter = iter(survivors)
    for dst_idx in range(dst_layers):
        if dst_idx in add_zero:
            mapping[dst_idx] = None
        else:
            try:
                mapping[dst_idx] = next(survivor_iter)
            except StopIteration:
                raise ValueError(
                    "Drop/add configuration did not provide enough surviving layers"
                )
    return mapping


LAYER_PREFIXES = [
    "core.blocks.",
    "context.pre_norms.",
    "context.context_sampler.",
    "context.context_bias_gen.",
]


def _remap_key_for_layers(name: str, mapping: dict[int, int | None]) -> str | None:
    for prefix in LAYER_PREFIXES:
        pos = name.find(prefix)
        if pos == -1:
            continue
        start = pos + len(prefix)
        end = name.find(".", start)
        if end == -1:
            continue
        idx = int(name[start:end])
        mapped = mapping.get(idx)
        if mapped is None:
            return None
        name = f"{name[:start]}{mapped}{name[end:]}"
    return name


def _copy_tensor_data(
    dst: torch.Tensor, src: torch.Tensor, *, allow_trim: bool
) -> torch.Tensor:
    if dst.shape == src.shape:
        return src.clone()
    if dst.ndim != src.ndim:
        raise ValueError("Cannot import parameters with different tensor ranks")
    slices = []
    for d, s in zip(dst.shape, src.shape):
        if d < s and not allow_trim:
            raise ValueError(
                "Destination parameter is smaller; rerun with --trim-model to allow trimming"
            )
        slices.append(slice(0, min(d, s)))
    result = dst.clone()
    result[tuple(slices)] = src[tuple(slices)]
    return result


def apply_imported_state(
    model: GRCEGPT,
    source_state: dict[str, torch.Tensor],
    *,
    allow_trim: bool,
    mapping: dict[int, int | None],
) -> None:
    dst_state = model.state_dict()
    new_state: dict[str, torch.Tensor] = {}
    for name, dst_tensor in dst_state.items():
        remapped = _remap_key_for_layers(name, mapping)
        if remapped is None:
            new_state[name] = dst_tensor
            continue
        src_tensor = source_state.get(remapped)
        if src_tensor is None:
            new_state[name] = dst_tensor
            continue
        new_state[name] = _copy_tensor_data(dst_tensor, src_tensor, allow_trim=allow_trim)
    model.load_state_dict(new_state)


@torch.no_grad()
def generate(
    model: GRCEGPT,
    idx: torch.Tensor,
    steps: int,
    *,
    suppress_newlines: bool = False,
    newline_token_id: int | None = None,
    think_settings: ThinkSettings | None = None,
    suppress_think: bool = False,
    suppress_think_prompt: bool = False,
    think_hard: bool = False,
    first_token_blocklist: Sequence[int] | None = None,
    sampling_strategy: str = "sample",
) -> tuple[torch.Tensor, int]:
    model.eval()
    idx = idx.clone()
    prompt_len = idx.size(1)
    think_token_id = active_think_token_id(think_settings)
    if not suppress_think and not suppress_think_prompt:
        if think_hard and think_settings is not None and think_settings.enabled:
            with torch.no_grad():
                logits, _, _ = model.forward_autoreg(
                    idx,
                    think_token_id=think_token_id,
                )
            prompt_ids = idx[0].tolist()
            think_token_id = think_settings.token_id
            new_tokens: list[int] = []
            for i, tok in enumerate(prompt_ids):
                new_tokens.append(tok)
                if think_token_id is None:
                    continue
                if i == 0:
                    new_tokens.append(think_token_id)
                    continue
                prev_logits = logits[0, i - 1]
                pred = int(torch.argmax(prev_logits).item())
                if pred != tok:
                    new_tokens.append(think_token_id)
            idx = torch.tensor([new_tokens], dtype=idx.dtype, device=idx.device)
            prompt_len = idx.size(1)
        else:
            idx = expand_prompt_with_thinking(model, idx, think_settings)
            prompt_len = idx.size(1)
    enforce_first_token_guard = bool(first_token_blocklist)
    blocklist = list(first_token_blocklist or [])
    for _ in range(steps):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _, _ = model.forward_autoreg(
            idx_cond,
            think_token_id=think_token_id,
        )
        logits_last = logits[:, -1, :]
        probs = F.softmax(logits_last, dim=-1)
        suppressed_ids: list[int] = []
        if suppress_newlines and newline_token_id is not None:
            suppressed_ids.append(int(newline_token_id))
        if suppress_think and think_settings is not None and think_settings.token_id is not None:
            suppressed_ids.append(int(think_settings.token_id))
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


# -----------------------------------------------------------------------------
# GRCE CLI Main Function
# -----------------------------------------------------------------------------

import signal
import traceback

def grce_main(args: argparse.Namespace) -> int:
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
    args.prompt = args.prompt.replace(THINK_SYMBOL, THINK_TOKEN)
    args.prompt = args.prompt.replace(UNDO_SYMBOL, UNDO_TOKEN)
    if args.think == 0 and THINK_TOKEN in args.prompt:
        raise ValueError(
            "Prompt contains thinking tokens but --no-think was specified. Remove them or omit --no-think."
        )
    if args.undo == 0 and UNDO_TOKEN in args.prompt:
        raise ValueError(
            "Prompt contains undo tokens but --undo is 0. Remove them or enable --undo."
        )
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
            args.n_layer = checkpoint_override_config.n_layer
            args.n_head = checkpoint_override_config.n_head
            args.n_embd = checkpoint_override_config.n_embd
            args.n_grce = checkpoint_override_config.n_grce
            args.n_xctx = checkpoint_override_config.n_xctx
            args.dropout = checkpoint_override_config.dropout
            args.detach_span = checkpoint_override_config.detach_span
            args.no_detach_ctx = not checkpoint_override_config.detach_context
            args.detach_layer = checkpoint_override_config.detach_layer
            args.tokenizer_vocab = checkpoint_override_config.vocab_size

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
        train_cache_path = data_dir / f"{args.corpus}_tokens_train_{args.tokenizer_vocab}.pt"
        test_cache_path = data_dir / f"{args.corpus}_tokens_test_{args.tokenizer_vocab}.pt"
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

        tokenizer_key = f"{args.corpus}_vocab_{args.tokenizer_vocab}"
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
        tok_wall_start = time.time()
        tok_cpu_start = time.process_time()
        vocab_source = full_train_text or ""
        reserved_tokens = 1 + len(GPT2TokenizerWrapper.EXTRA_SPECIAL_TOKENS)
        if args.tokenizer_vocab <= reserved_tokens:
            raise ValueError(
                f"--tokenizer-vocab must exceed reserved tokens ({reserved_tokens}); got {args.tokenizer_vocab}"
            )
        target_vocab = max(0, args.tokenizer_vocab - reserved_tokens)
        tokenizer = GPT2TokenizerWrapper(
            vocab_source,
            tokenizer_path,
            target_vocab,
            pretrained_json=tokenizer_json,
        )
        expected_vocab_size = args.tokenizer_vocab
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

        think_settings = ThinkSettings(
            max_steps=args.think,
            token_id=tokenizer.think_id,
        )
        undo_settings = UndoSettings(
            max_pairs=args.undo,
            token_id=tokenizer.undo_id,
            fill_choices=[],
        )

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

        def ensure_token_absent(tensor: torch.Tensor, token_id: int, label: str, enabled: bool) -> None:
            if enabled or token_id is None:
                return
            if (tensor == token_id).any().item():
                raise ValueError(
                    f"Training data includes {label} token but {label} mode is disabled."
                )

        ensure_token_absent(train_tokens, tokenizer.think_id, "think", args.think > 0)
        ensure_token_absent(test_tokens, tokenizer.think_id, "think", args.think > 0)
        ensure_token_absent(train_tokens, tokenizer.undo_id, "undo", args.undo > 0)
        ensure_token_absent(test_tokens, tokenizer.undo_id, "undo", args.undo > 0)

        if train_text is None and train_bytes == 0:
            train_bytes = len(train_tokens)  # fallback when text absent
        if test_text is None and test_bytes == 0:
            test_bytes = len(test_tokens)

        train_token_count = int(train_tokens.numel())
        test_token_count = int(test_tokens.numel())
        print(
            color_text(
                f"Dataset size: {train_token_count:,} train tokens, {test_token_count:,} test tokens",
                Colors.CYAN,
            )
        )
        tok_summary = (
            f"[tokenizer] wall={time.time()-tok_wall_start:.2f}s cpu={time.process_time()-tok_cpu_start:.2f}s\n"
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
                colored = color_tokens(
                    tokenizer,
                    chunk_tokens,
                    [Colors.MAGENTA, Colors.GREEN],
                    bold=False,
                    think_token_id=tokenizer.think_id,
                    undo_token_id=tokenizer.undo_id,
                )
                print(colored)

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
        if selected_action == "size":
            describe_model_size(
                config,
                args.block_size,
                check=getattr(args, "check", False),
                estimate=getattr(args, "estimate", False),
            )
            return
        model_tag = build_model_tag(config)
        if args.think > 0:
            model_tag += "_think"
        if args.undo > 0:
            model_tag += f"_undo{args.undo}"
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
        temp_model = GRCEGPT(config)
        non_emb_params = sum(
            p.numel()
            for name, p in temp_model.named_parameters()
            if p.requires_grad and "tok_emb" not in name and "pos_emb" not in name
        )
        print(f"Trainable model params (excl. embeddings): {non_emb_params:,}")
        tok_vecs = temp_model.core.tok_emb.num_embeddings
        pos_vecs = temp_model.core.pos_emb.num_embeddings
        emb_vectors = tok_vecs + pos_vecs
        emb_params = temp_model.core.tok_emb.weight.numel() + temp_model.core.pos_emb.weight.numel()
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
            import_wall = time.time()
            import_cpu = time.process_time()
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
            write_wall_start = time.time()
            write_cpu_start = time.process_time()
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
            write_wall = time.time() - write_wall_start
            write_cpu = time.process_time() - write_cpu_start
            import_wall = time.time() - import_wall
            import_cpu = time.process_time() - import_cpu
            print(
                color_text(
                    f"[import] total steps: {total_steps}; time spent: wall={import_wall:.2f}s cpu={import_cpu:.2f}s; writing model: wall={write_wall:.2f}s cpu={write_cpu:.2f}s",
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
                think_settings=think_settings,
                suppress_think=args.no_think_output,
                suppress_think_prompt=args.no_think_prompt,
                think_hard=args.think_hard,
                underline_tokens=args.underline,
                default_prompt_boundary=default_prompt_boundary,
                boundary_blocklist=boundary_blocklist,
                show_train_loss_details=args.train_loss_details,
                show_test_loss_details=not args.no_test_loss_details,
            )
            return

        if selected_action == "test":
            run_test_slice(
                dataset=dataset,
                tokenizer=tokenizer,
                model=model,
                block_size=args.block_size,
                start_pos=args.test_start,
                think_settings=think_settings,
                underline_tokens=args.underline,
            )
            return

        reward_scale = 0.0
        if args.reward_relu > 0:
            reward_scale = 10 ** (-float(args.reward_relu))
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
            if args.think > 0:
                plus_tags.append("+THINK")
            else:
                minus_tags.append(" wo/THINK")
            if args.undo > 0:
                plus_tags.append("+UNDO")
            else:
                minus_tags.append(" wo/UNDO")
            label = "".join(tags + plus_tags + minus_tags)
            hours = total_train_wall / 3600.0
            days = hours / 24.0
            train_chars_cycle = (args.block_size + 1) * args.batch_size * args.steps
            test_chars_cycle = (
                (args.block_size + 1)
                * args.batch_size
                * max(1, args.eval_iters * count_eval_calls(args.steps, args.eval_interval))
            )
            train_start = int(dataset.positions.get("train", 0))
            test_start = int(dataset.positions.get("test", 0))
            dataset.prepare_cycle("train", train_chars_cycle)
            dataset.prepare_cycle("test", test_chars_cycle)

            train_chunk = dataset.chunks.get("train")
            test_chunk = dataset.chunks.get("test")

            def format_range(start: int, span: int) -> str:
                if span <= 0:
                    return f"{start}-{start}"
                end = start + span - 1
                return f"{start}-{end}"

            train_span = int(train_chunk.size(0)) if train_chunk is not None else 0
            test_span = int(test_chunk.size(0)) if test_chunk is not None else 0
            train_range = format_range(train_start, train_span)
            test_range = format_range(test_start, test_span)
            print(
                color_text(
                    f"\nDataset: train tokens {train_range}, test tokens {test_range}",
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
                preeval_wall_total,
                preeval_cpu_total,
            ) = train_model(
                model,
                dataset,
                device,
                args.steps,
                args.block_size,
                args.batch_size,
                args.eval_interval,
                args.eval_iters,
                total_steps,
                prompt_tokens,
                args.generate,
                tokenizer,
                suppress_newlines=args.no_newlines,
                newline_token_id=newline_token_id,
                think_settings=think_settings,
                suppress_think_output=args.no_think_output,
                suppress_think_prompt=args.no_think_prompt,
                think_hard=args.think_hard,
                undo_settings=undo_settings,
                prompt_tracker=prompt_tracker,
                reset_prompt_queue=args.reset_prompt_each_cycle,
                reward_relu=reward_scale,
                context_dropout_interval=args.context_dropout_interval,
                cycle_wall_start=cycle_wall,
                base_wall_seconds=total_train_wall,
                show_time=args.time,
                underline_tokens=args.underline,
                default_prompt_boundary=default_prompt_boundary,
                boundary_blocklist=boundary_blocklist,
                show_train_loss_details=args.train_loss_details,
                show_test_loss_details=not args.no_test_loss_details,
                long_loss_log=args.long_loss_log,
                full_eval_stride=args.eval_full,
            )
            loss_history.extend(updates)

            train_wall = cycle_wall_elapsed
            train_cpu = cycle_cpu_elapsed
            eval_wall = eval_wall_total
            eval_cpu = eval_cpu_total
            pure_train_wall = max(0.0, train_wall - eval_wall)
            pure_train_cpu = max(0.0, train_cpu - eval_cpu)
            preeval_wall_ratio = (
                preeval_wall_total / train_wall if train_wall > 0 else 0.0
            )
            preeval_cpu_ratio = (
                preeval_cpu_total / train_cpu if train_cpu > 0 else 0.0
            )
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
            if log_file is not None:
                log_file.flush()
            if ansi_file is not None:
                ansi_file.flush()
            cycle_part = color_text(f"[cycle {cycle}]", Colors.CYAN)
            train_part = color_text(
                f" train: wall={pure_train_wall:.2f}s cpu={pure_train_cpu:.2f}s;",
                Colors.MAGENTA,
            )
            eval_part = color_text(
                f" eval: wall={eval_wall:.2f}s cpu={eval_cpu:.2f}s;",
                Colors.GREEN,
            )
            ratio_part = color_text(
                f" train-pre-eval/train-total: wall={preeval_wall_ratio:.2f} cpu={preeval_cpu_ratio:.2f};",
                Colors.CYAN,
            )
            updated_part = color_text(" model updated.", Colors.YELLOW)
            print(cycle_part + train_part + eval_part + ratio_part + updated_part)
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

    except KeyboardInterrupt:
        if args.debug_interrupt:
            raise
        traceback.print_exc()
        print(color_text("Interrupted by user; exiting cleanly.", Colors.RED, bold=True))
    except TimeoutAlarm:
        traceback.print_exc()
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
    sys.exit(grce_main(cli_args))
