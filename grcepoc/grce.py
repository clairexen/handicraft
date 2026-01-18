"""GRCE proof-of-concept. Most of it is written by ChatGPT/Codex. I told
it to base it loosely the picoGPT.

This script keeps the picoGPT spirit of being small and hackable while
adding the Gradient-limited Recurrent Context Encoding (GRCE) channel described in
the README. It trains a tiny GPT-style tokenizer-backed Transformer on the bundled
Simple English Wikipedia split and shows how the recurrent context vector can
be integrated with a configurable gradient-limiting constraint across time.
"""
from __future__ import annotations

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
from typing import Dict, List, Tuple

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

HF_CACHE_DIR = pathlib.Path(".cache_transformers")
HF_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR.resolve()))

from transformers import GPT2TokenizerFast

FANCY_SPACE = "\u2423"  # Open Box symbol for visible spaces
FANCY_ENTER = "\u23CE"  # Return symbol for visible newlines
THINK_TOKEN = "<think>"
THINK_SYMBOL = "\u2754"  # white question mark
# THINK_SYMBOL = "\u21BA"  # anticlockwise circle arrow (alternative option)
UNDO_TOKEN = "<undo>"
UNDO_SYMBOL = "\u21A9"  # leftwards arrow with hook
ASCII_LETTERS = set(string.ascii_letters)

PROMPT_GOALS = [
    ("ice", " cold"),
    ("one plus one is", " two"),
    ("the first letter of the alphabet is", " a"),
    ("the color of a red apple is", " red"),
    ("the opposite of hot is", " cold"),
    ("water freezes at", " 0"),
    ("sun rises in the", " east"),
    ("earth's satellite is the", " moon"),
    ("a baby cat is called a", " kitten"),
]


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
# Data utilities (borrow the spirit of picoGPT's Shakespeare example)
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
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    WHITE = "\033[97m"


def color_text(text: str, color: str, *, bold: bool = False) -> str:
    prefix = Colors.BOLD if bold else ""
    return f"{prefix}{color}{text}{Colors.RESET}"


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
    def __init__(
        self,
        train_text: str,
        cache_path: pathlib.Path,
        vocab_size: int,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_path
        self.extra_special_tokens: list[str] = []
        self.extra_special_tokens.append(THINK_TOKEN)
        self.extra_special_tokens.append(UNDO_TOKEN)
        self.tokenizer = self._load_or_train(
            train_text,
            cache_path,
            vocab_size,
            self.extra_special_tokens,
        )
        self.vocab_size = len(self.tokenizer)
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
        tokenizer = Tokenizer(BPE(unk_token=None))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        byte_values = sorted(set(train_text.encode("utf-8")))
        initial_alphabet = [chr(b) for b in byte_values] or ByteLevel.alphabet()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
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
        self.completed: list[bool] = []
        self._cache: dict[int, torch.Tensor] = {}
        self._expected_token_ids: dict[int, List[int]] = {}
        self.load_state(state)

    def load_state(self, state: dict | None) -> None:
        if state and isinstance(state.get("completed"), list):
            raw = state.get("completed", [])
            self.completed = [bool(val) for val in raw][: len(PROMPT_GOALS)]
        if len(self.completed) != len(PROMPT_GOALS):
            self.completed = [False] * len(PROMPT_GOALS)

    def serialize(self) -> dict:
        return {"completed": list(self.completed)}

    def next_goal(self) -> tuple[int | None, tuple[str, str] | None]:
        for idx, done in enumerate(self.completed):
            if not done:
                return idx, PROMPT_GOALS[idx]
        return None, None

    def prompt_tensor(self, idx: int, device: torch.device) -> torch.Tensor:
        if idx not in self._cache:
            tensor = self.tokenizer.encode(PROMPT_GOALS[idx][0]).unsqueeze(0)
            self._cache[idx] = tensor
        return self._cache[idx].to(device)

    def expected_text(self, idx: int) -> str:
        return PROMPT_GOALS[idx][1]
    
    def expected_token_ids(self, idx: int) -> List[int]:
        if idx not in self._expected_token_ids:
            tensor = self.tokenizer.encode(self.expected_text(idx))
            self._expected_token_ids[idx] = tensor.tolist()
        return self._expected_token_ids[idx]

    def mark_if_satisfied(self, idx: int, completion_ids: List[int]) -> bool:
        if idx is None or idx < 0 or idx >= len(self.completed):
            return False
        expected_ids = self.expected_token_ids(idx)
        if len(completion_ids) < len(expected_ids):
            return False
        if completion_ids[: len(expected_ids)] == expected_ids and not self.completed[idx]:
            self.completed[idx] = True
            return True
        return False

    def remaining(self) -> int:
        return sum(1 for done in self.completed if not done)

    def is_completed(self, idx: int | None) -> bool:
        return idx is None or self.completed[idx]

    def pending_indices(self, limit: int | None = None) -> List[int]:
        indices = [i for i, done in enumerate(self.completed) if not done]
        if limit is not None:
            return indices[: max(0, int(limit))]
        return indices


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
            )
    if prev_mode and need_logits:
        model.train()
    think_scores = None
    if think_enabled:
        think_labels = torch.full_like(inputs, -1)
        think_slot_mask = torch.zeros_like(inputs, dtype=torch.bool, device=device)
        if full_logits is not None and think is not None and think.token_id is not None:
            think_scores = full_logits[..., think.token_id]
    for row in range(B):
        row_think_active = think_enabled and row not in think_disabled_rows and row not in forced_context_off
        max_insert_budget = block_size - 1
        think_cap = think.max_steps if row_think_active else 0
        think_cap = min(max_insert_budget, think_cap)
        think_count = random.randint(0, think_cap) if think_cap > 0 else 0
        remaining_budget = max_insert_budget - think_count
        undo_cap = undo.max_pairs if undo_enabled else 0
        undo_cap = min(undo_cap, remaining_budget // 2)
        undo_pairs = random.randint(0, undo_cap) if undo_cap > 0 else 0
        max_think_allowed = (block_size - 2 * undo_pairs) // 2
        if think_count > max_think_allowed:
            think_count = max_think_allowed
        keep_len = block_size - think_count - 2 * undo_pairs
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
        seq_entries.append({"token": tail_token, "tag": "base", "base_index": None})

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

        if row_think_active and think_count > 0 and think_scores is not None:
            scores = think_scores[row, :keep_len]
            if scores.numel() > 0:
                picks = min(think_count + 1, scores.numel())
                topk = torch.topk(scores, picks).indices.tolist()
                selected_positions = topk[:]
                drop_total = min(2, len(selected_positions))
                if drop_total > 0:
                    drop_choices = sorted(
                        random.sample(range(len(selected_positions)), k=drop_total), reverse=True
                    )
                    for idx in drop_choices:
                        selected_positions.pop(idx)
                selected_positions = selected_positions[:think_count]
                needed = max(0, think_count - len(selected_positions))
                if needed > 0:
                    selected_set = set(selected_positions)
                    remaining_positions = [
                        pos for pos in range(keep_len) if pos not in selected_set
                    ]
                    random.shuffle(remaining_positions)
                    selected_positions.extend(remaining_positions[:needed])
                random.shuffle(selected_positions)
                for pos in selected_positions:
                    insert_idx = None
                    for idx, entry in enumerate(seq_entries):
                        if entry.get("base_index") == pos:
                            insert_idx = idx
                            break
                    if insert_idx is None:
                        insert_idx = len(seq_entries) - 1
                    seq_entries.insert(
                        insert_idx,
                        {
                            "token": int(think.token_id),
                            "tag": "think",
                            "base_index": None,
                        },
                    )

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
        if think_labels is not None:
            for idx, entry in enumerate(seq_entries[:-1]):
                if entry.get("tag") == "think":
                    label = seq_entries[idx + 1]["token"]
                    think_labels[row, idx] = int(label)
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



def load_or_prepare_tokens(
    split: str,
    text_path: str,
    text: str | None,
    limit: int,
    cache_path: pathlib.Path,
    tokenizer: GPT2TokenizerWrapper,
    seed: int,
) -> Tuple[torch.Tensor, str | None, int, int]:
    if cache_path.exists():
        payload = torch.load(cache_path)
        tokens = payload["tokens"].long()
        bytes_count = int(payload.get("bytes", 0))
        inserts = int(payload.get("inserts", 0))
        trimmed_text = None
        if text is not None:
            trimmed_text = text if limit <= 0 else text[:limit]
        print(color_text(f"Loaded cached {split} tokens from {cache_path}", Colors.YELLOW))
        return tokens, trimmed_text, bytes_count, inserts

    print(color_text(f"Tokenizing raw {split} data: {text_path}...", Colors.BLUE))

    if text is None:
        raise FileNotFoundError(
            f"No cached tokens at {cache_path} and source text missing for {split}."
        )
    trimmed_text = text if limit <= 0 else text[:limit]
    if not trimmed_text:
        raise ValueError(f"Text for {split} split is empty after applying character limit")
    tokens = tokenizer.encode_corpus(trimmed_text)
    bytes_count = len(trimmed_text.encode("utf-8"))
    inserts = 0
    torch.save({"tokens": tokens, "bytes": bytes_count, "inserts": inserts}, cache_path)
    print(color_text(f"Saved {split} token cache to {cache_path}", Colors.YELLOW))
    return tokens, trimmed_text, bytes_count, inserts


# -----------------------------------------------------------------------------
# Model components (picoGPT-style Transformer blocks + GRCE channel)
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    vocab_size: int = 2000  # GPT-2 base supports ~50k merges; we stay small for the PoC.
    block_size: int = 64    # GPT-2 base uses 1024 tokens.
    n_layer: int = 6        # GPT-2 base uses 12 layers.
    n_head: int = 4         # GPT-2 base uses 12 attention heads.
    n_embd: int = 192       # GPT-2 base uses 768 embedding dims.
    n_grce: int = 64        # GRCE context dims.
    dropout: float = 0.05
    context_span: int = 2   # Detach gradients every N positions (0 disables detaching).
    context_dropout: int = 0  # Target number of GRCE dropouts per block (0 disables).
    grce_layered: bool = False


MODEL_CONFIG_TEMPLATE = ModelConfig()


def describe_model_size(config: ModelConfig) -> None:
    model = GRCEGPT(config)
    categories = {
        "token_embeddings": 0,
        "position_embeddings": 0,
        "core": 0,
        "context": 0,
    }
    layer_counts = [0] * config.n_layer
    context_parts = {
        "samplers": 0,
        "mlp": 0,
        "bias": 0,
        "norm": 0,
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        size = param.numel()
        if name.startswith("core.tok_emb"):
            categories["token_embeddings"] += size
        elif name.startswith("core.pos_emb"):
            categories["position_embeddings"] += size
        elif name.startswith("context"):
            categories["context"] += size
            if ".context_sampler" in name:
                context_parts["samplers"] += size
            elif ".context_mlp" in name:
                context_parts["mlp"] += size
            elif ".context_bias_gen" in name:
                context_parts["bias"] += size
            elif ".context_norm" in name:
                context_parts["norm"] += size
        else:
            categories["core"] += size
            if ".blocks." in name:
                try:
                    idx = int(name.split("blocks.")[1].split(".")[0])
                    if 0 <= idx < len(layer_counts):
                        layer_counts[idx] += size
                except ValueError:
                    pass
    total = sum(categories.values())
    print(
        f"\nTransformer stack breakdown ({config.n_layer} layers):"
    )
    for idx, size in enumerate(layer_counts):
        mb = size * 4 / 1_000_000
        print(f"  layer {idx:02d}: {size:>12,d} params ({mb:.2f} MB)")
    print("\nGRCE resource breakdown:")
    for label, size in context_parts.items():
        mb = size * 4 / 1_000_000
        print(f"  {label:10s}: {size:>12,d} params ({mb:.2f} MB)")
    print("\nModel parameter breakdown:")
    for label, size in categories.items():
        mb = size * 4 / 1_000_000
        pct = (size / total * 100) if total else 0
        print(f"  {label:22s}: {size:>12,d} params ({mb:.2f} MB, {pct:.1f}%)")
    if total:
        print(f"  {'total':22s}: {total:>12,d} params ({total*4/1_000_000:.2f} MB)")


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
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
        self, x: torch.Tensor, *, record_mask: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = x + self.attn(self.ln1(x))
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

    def forward(
        self,
        idx: torch.Tensor,
        block_biases: List[torch.Tensor] | None = None,
        *,
        record_relu_mask: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor] | None]:
        B, T = idx.shape
        device = idx.device
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = self.drop(tok + pos)
        block_inputs: List[torch.Tensor] = []
        relu_masks: List[torch.Tensor | None] | None = None
        if record_relu_mask:
            relu_masks = [None] * len(self.blocks)
        for layer_idx, block in enumerate(self.blocks):
            if block_biases is not None:
                x = x + block_biases[layer_idx]
            block_inputs.append(x[:, -1, :])
            x, layer_mask = block(x, record_mask=record_relu_mask)
            if record_relu_mask and relu_masks is not None and layer_mask is not None:
                relu_masks[layer_idx] = layer_mask
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, block_inputs, relu_masks


class GRCEContextChannel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.disabled = config.n_grce <= 0
        self.config = config
        self.context_span = max(0, int(config.context_span))
        self.context_dim = config.n_grce
        self.context_dropout = max(0, int(config.context_dropout))
        self.layered = config.grce_layered
        if self.layered:
            if config.n_layer <= 0 or config.n_grce % config.n_layer != 0:
                raise ValueError("--grce-layered requires n_grce to be divisible by n_layer")
            self.layer_chunk = config.n_grce // config.n_layer
        else:
            self.layer_chunk = None
        if not self.disabled:
            mid = 2 * config.n_grce if self.layered else 4 * config.n_grce
            self.pre_norms = nn.ModuleList(
                nn.LayerNorm(config.n_embd) for _ in range(config.n_layer)
            )
            self.context_sampler = nn.ModuleList(
                [self._build_sampler(config.n_embd, config.n_grce) for _ in range(config.n_layer)]
            )
            self.context_mlp = nn.Sequential(
                nn.Linear(config.n_grce, mid),
                nn.ReLU(),
                nn.Linear(mid, config.n_grce),
            )
            self.context_norm = nn.LayerNorm(config.n_grce)
            self.context_bias_gen = nn.ModuleList(
                [self._build_bias(config.n_grce, config.n_embd) for _ in range(config.n_layer)]
            )

    def _build_sampler(self, in_dim: int, out_dim: int) -> nn.Module:
        if not self.layered or self.layer_chunk is None:
            return nn.Linear(in_dim, out_dim)
        chunk = self.layer_chunk
        return nn.Sequential(
            nn.Linear(in_dim, chunk),
            nn.Linear(chunk, out_dim),
        )

    def _build_bias(self, in_dim: int, out_dim: int) -> nn.Module:
        if not self.layered or self.layer_chunk is None:
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
            residual = prev_context.detach() if stop_grad else prev_context
            fused = fused + residual
        context = self.context_mlp(fused)
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
        self.context = GRCEContextChannel(config)

    def forward_autoreg(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        disable_context: bool = False,
        capture_activations: bool = False,
        think_token_id: int | None = None,
        collect_relu_mask: bool = False,
        disable_context_dropout: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        B, T = idx.shape
        device = idx.device
        use_context = not self.context.disabled and not disable_context
        context = None
        if use_context:
            context_dim = self.context.context_dim
            context = torch.zeros(B, context_dim, device=device)
        logits_steps = []
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
        if capture_activations:
            activation_store = {
                "block_norms": [[] for _ in range(self.config.n_layer)],
                "context_norms": [],
            }
        for t in range(T):
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
            if use_context and context is not None:
                bias_vectors = self.context.project(context)
                block_biases = []
                for bias_vec in bias_vectors:
                    full = torch.zeros(
                        B,
                        prefix.size(1),
                        self.config.n_embd,
                        device=device,
                        dtype=bias_vec.dtype,
                    )
                    full[:, -1, :] = bias_vec
                    block_biases.append(full)
            logits, block_inputs, layer_masks = self.core(
                prefix,
                block_biases=block_biases,
                record_relu_mask=collect_relu_mask,
            )
            if relu_activity is not None:
                relu_activity.append(layer_masks)
            if activation_store is not None:
                for layer_idx, block_inp in enumerate(block_inputs):
                    norms = torch.linalg.vector_norm(block_inp.detach(), dim=-1)
                    activation_store["block_norms"][layer_idx].extend(
                        norms.cpu().tolist()
                    )
            if use_context and context is not None:
                span = self.context.context_span
                if span <= 0:
                    stop_grad = False
                elif span == 1:
                    stop_grad = True
                else:
                    stop_grad = (t % span == 0)
                use_grce = True
                if self.training and not disable_context_dropout:
                    seq_len = max(1, self.config.block_size)
                    drop_prob = min(1.0, 1 / seq_len)
                    if random.random() < drop_prob:
                        use_grce = False
                if use_grce:
                    context, raw_context = self.context.update(
                        block_inputs,
                        prev_context=context,
                        stop_grad=stop_grad,
                    )
                    if activation_store is not None:
                        ctx_norms = torch.linalg.vector_norm(raw_context.detach(), dim=-1)
                        activation_store["context_norms"].extend(ctx_norms.cpu().tolist())
            logits_steps.append(logits[:, -1:, :])
        logits = torch.cat(logits_steps, dim=1)
        if relu_activity is not None:
            if activation_store is None and need_store:
                activation_store = {}
            if activation_store is not None:
                activation_store.setdefault("relu_activity", relu_activity)
        return logits, context, activation_store


def build_model_tag(config: ModelConfig) -> str:
    ctx_prefix = "xctx" if config.grce_layered else "ctx"
    tag = (
        f"v{config.vocab_size}_bs{config.block_size}_emb{config.n_embd}_"
        f"layers{config.n_layer}_heads{config.n_head}_{ctx_prefix}{config.n_grce}"
    )
    return tag


def compute_think_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    think_token_id: int | None,
    plan_required: torch.Tensor | None = None,
) -> torch.Tensor:
    if think_token_id is None:
        return logits.new_tensor(0.0)
    mask = targets == think_token_id
    if not mask.any():
        return logits.new_tensor(0.0)
    probs = F.softmax(logits, dim=-1)
    preds = torch.argmax(logits, dim=-1)
    entries: list[torch.Tensor] = []
    labels: list[float] = []
    B, T = targets.shape
    for b in range(B):
        think_positions = torch.nonzero(mask[b], as_tuple=False).flatten()
        if think_positions.numel() == 0:
            continue
        for pos_tensor in think_positions:
            pos = int(pos_tensor.item())
            next_idx = pos + 1
            while next_idx < T and targets[b, next_idx] == think_token_id:
                next_idx += 1
            label = 0.0
            if next_idx < T:
                label = 1.0 if preds[b, next_idx] == targets[b, next_idx] else 0.0
            if plan_required is not None:
                plan_needed = bool(plan_required[b, pos_tensor.item()])
                if not plan_needed:
                    label = 0.0
            entries.append(probs[b, pos, think_token_id])
            labels.append(label)
    if not entries:
        return logits.new_tensor(0.0)
    prob_tensor = torch.stack(entries)
    label_tensor = prob_tensor.new_tensor(labels)
    return F.binary_cross_entropy(prob_tensor, label_tensor)


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
    while pos < idx.size(1) and inserted < max_insertions:
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
            continue
        pos += 1
    return idx


# -----------------------------------------------------------------------------
# Training / generation helpers
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
    think_settings: ThinkSettings | None = None,
    undo_settings: UndoSettings | None = None,
    batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[float, float]:
    ce_losses = []
    learned_losses = []
    if batches is None:
        batches = [
            dataset.get_batch(split, block_size, batch_size, device) for _ in range(iters)
        ]
    think_token_id = active_think_token_id(think_settings)
    for xb, yb in batches:
        aug_xb, aug_yb, random_mask, think_labels, think_slot_mask = augment_training_batch(
            model,
            xb,
            yb,
            think_settings,
            undo_settings,
        )
        logits, _, _ = model.forward_autoreg(
            aug_xb,
            disable_context=disable_context,
            think_token_id=think_token_id,
        )
        logits = apply_think_slot_mask(logits, think_slot_mask, think_settings)
        loss_targets = build_loss_targets(aug_yb, think_settings, random_mask)
        main_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            loss_targets.view(-1),
            ignore_index=LOSS_IGNORE_INDEX,
        )
        total_loss = main_loss
        if think_settings is not None and think_settings.enabled:
            plan_required = None
            if think_labels is not None and think_settings.token_id is not None:
                mask = think_labels >= 0
                if mask.any():
                    plan_logits = logits.clone()
                    plan_logits[..., think_settings.token_id] = -1e9
                    selected_logits = plan_logits[mask]
                    targets_plan = think_labels[mask]
                    plan_loss = F.cross_entropy(
                        selected_logits,
                        targets_plan,
                    )
                    total_loss = total_loss + plan_loss
                    plan_required = torch.zeros_like(think_labels, dtype=torch.bool)
                    plan_pred = torch.argmax(selected_logits, dim=-1)
                    plan_required[mask] = plan_pred != targets_plan
            total_loss = total_loss + compute_think_penalty(
                logits,
                aug_yb,
                think_settings.token_id,
                plan_required=plan_required,
            )
        ce_losses.append(main_loss.item())
        learned_losses.append(total_loss.item())
    ce_avg = sum(ce_losses) / len(ce_losses)
    learned_avg = sum(learned_losses) / len(learned_losses)
    return ce_avg, learned_avg


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
    *,
    reward_relu: bool = False,
    nogrce_interval: int = 1,
    cycle_wall_start: float,
    base_wall_seconds: float,
    cycle_prompt_indices: List[int] | None = None,
) -> Tuple[int, List[Dict[str, float]]]:
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    total_steps = start_step
    history_updates: List[Dict[str, float]] = []
    printed_header = False
    think_enabled = think_settings is not None and think_settings.enabled
    undo_enabled = undo_settings is not None and undo_settings.enabled
    think_token_id = active_think_token_id(think_settings)
    reward_tracker = (
        ReLURewardTracker(model, steps, scale=reward_relu)
        if reward_relu > 0
        else None
    )
    prompt_queue = list(cycle_prompt_indices or [])
    show_think_columns = think_enabled
    show_target_headers = think_enabled or undo_enabled
    nogrce_interval = max(0, int(nogrce_interval))
    for step in range(1, steps + 1):
        xb, yb = dataset.get_batch("train", block_size, batch_size, device)
        current_step_index = total_steps
        nogrce_active = nogrce_interval > 0 and current_step_index % nogrce_interval == 0
        disable_rows: set[int] = set()
        if nogrce_active:
            drop_target = 1
            drop_prob = min(1.0, drop_target / max(1, batch_size))
            for row_idx in range(batch_size):
                if random.random() < drop_prob:
                    disable_rows.add(row_idx)
        think_disabled_rows = set()
        if think_enabled and batch_size > 0:
            think_disabled_rows.add(random.randrange(batch_size))
        xb, yb, random_mask, think_labels, think_slot_mask = augment_training_batch(
            model,
            xb,
            yb,
            think_settings,
            undo_settings,
            disable_context_rows=disable_rows,
            disable_think_rows=think_disabled_rows,
        )
        logits, _, activation_store = model.forward_autoreg(
            xb,
            yb,
            think_token_id=think_token_id,
            collect_relu_mask=reward_tracker is not None,
            disable_context_dropout=not nogrce_active,
        )
        logits = apply_think_slot_mask(logits, think_slot_mask, think_settings)
        logits_flat = logits.view(-1, logits.size(-1))
        loss_targets = build_loss_targets(yb, think_settings, random_mask)
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
        plan_required = None
        think_loss = logits.new_tensor(0.0)
        if think_settings is not None and think_settings.enabled:
            think_loss = compute_think_penalty(
                logits,
                yb,
                think_settings.token_id,
                plan_required=plan_required,
            )
        plan_loss = logits.new_tensor(0.0)
        if think_labels is not None and think_settings is not None and think_settings.token_id is not None:
            mask = think_labels >= 0
            if mask.any():
                plan_logits = logits.clone()
                plan_logits[..., think_settings.token_id] = -1e9
                selected_logits = plan_logits[mask]
                targets_plan = think_labels[mask]
                plan_loss = F.cross_entropy(
                    selected_logits,
                    targets_plan,
                )
                plan_pred = torch.argmax(selected_logits, dim=-1)
                plan_required = torch.zeros_like(think_labels, dtype=torch.bool)
                plan_required[mask] = plan_pred != targets_plan
        loss = main_loss + think_loss + plan_loss
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

        if step == 1 or step % eval_interval == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                split_metrics: dict[str, dict[str, float]] = {}
                cached_batches: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
                for split in ("train", "test"):
                    cached_batches[split] = [
                        dataset.get_batch(split, block_size, batch_size, device)
                        for _ in range(eval_iters)
                    ]
                    for suffix, disable in (("", False), ("_nogrce", True)):
                        ce_loss, learned_loss = evaluate_split(
                            model,
                            dataset,
                            device,
                            block_size,
                            batch_size,
                            split,
                            eval_iters,
                            disable_context=disable,
                            think_settings=think_settings,
                            undo_settings=undo_settings,
                            batches=cached_batches[split],
                        )
                        split_metrics[f"{split}{suffix}"] = {
                            "ce": float(ce_loss),
                            "learned": float(learned_loss),
                        }
                    if think_settings is not None and think_settings.enabled:
                        ce_loss, learned_loss = evaluate_split(
                            model,
                            dataset,
                            device,
                            block_size,
                            batch_size,
                            split,
                            eval_iters,
                            disable_context=False,
                            think_settings=None,
                            undo_settings=None,
                            batches=cached_batches[split],
                        )
                        split_metrics[f"{split}_nothink"] = {
                            "ce": float(ce_loss),
                            "learned": float(learned_loss),
                        }
                prompt_input = sample_prompt
                current_prompt_idx = None
                if prompt_tracker is not None and prompt_queue:
                    while prompt_queue and prompt_tracker.is_completed(prompt_queue[0]):
                        prompt_queue.pop(0)
                    if prompt_queue:
                        current_prompt_idx = prompt_queue.pop(0)
                        prompt_input = prompt_tracker.prompt_tensor(current_prompt_idx, device)
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
                )
            model.train()
            sample_ids = sample_tokens[0].detach().cpu().tolist()
            prompt_ids = sample_ids[:prompt_len]
            completion_ids = sample_ids[prompt_len:]

            if prompt_tracker is not None and current_prompt_idx is not None:
                if prompt_tracker.mark_if_satisfied(current_prompt_idx, completion_ids):
                    expected = prompt_tracker.expected_text(current_prompt_idx)
                    prompt_text = PROMPT_GOALS[current_prompt_idx][0]
                    print(
                        color_text(
                            f"Prompt #{current_prompt_idx + 1} satisfied: {prompt_text} (expected '{expected}')",
                            Colors.YELLOW,
                            bold=True,
                        )
                    )

            prefix_text = color_tokens(
                tokenizer,
                prompt_ids,
                [Colors.MAGENTA, Colors.GREEN],
                bold=False,
                think_token_id=tokenizer.think_id,
                undo_token_id=tokenizer.undo_id,
            )
            completion_text = color_tokens(
                tokenizer,
                completion_ids,
                [Colors.YELLOW, Colors.CYAN],
                think_token_id=tokenizer.think_id,
                undo_token_id=tokenizer.undo_id,
            )
            colored_sample = prefix_text + completion_text
            if not printed_header:
                if show_target_headers:
                    train_header = "train loss (target)  nogrce (target)"
                    test_header = "test loss (target)  nogrce (target)"
                else:
                    train_header = "train loss  nogrce"
                    test_header = "test loss  nogrce"
                if show_think_columns:
                    train_header += "  plain"
                    test_header += "  plain"
                remaining = prompt_tracker.remaining() if prompt_tracker else 0
                total_prompts = len(PROMPT_GOALS)
                header_line = (
                    color_text("step", Colors.CYAN)
                    + " | "
                    + color_text(train_header, Colors.GREEN)
                    + " | "
                    + color_text(test_header, Colors.MAGENTA)
                    + color_text(f" | sample ({total_prompts - remaining}/{total_prompts})", Colors.YELLOW)
                )
                print(header_line)
                printed_header = True

            hidden_target_warning_emitted = False

            def format_metric(key: str, *, include_target: bool = True) -> str:
                nonlocal hidden_target_warning_emitted
                metric = split_metrics[key]
                ce_val = metric["ce"]
                target_val = metric["learned"]
                differs = abs(target_val - ce_val) >= 1e-6
                if (
                    not show_target_headers
                    and differs
                    and not hidden_target_warning_emitted
                    and include_target
                ):
                    print(
                        color_text(
                            "warning: target values differ but target columns are hidden",
                            Colors.YELLOW,
                        )
                    )
                    hidden_target_warning_emitted = True
                if show_target_headers and include_target:
                    return f"{ce_val:.2f} ({target_val:.2f})"
                return f"{ce_val:.2f}"

            train_parts = [
                format_metric("train"),
                format_metric("train_nogrce"),
            ]
            if show_think_columns:
                train_parts.append(format_metric("train_nothink", include_target=False))
            train_values = "  ".join(train_parts)

            test_parts = [
                format_metric("test"),
                format_metric("test_nogrce"),
            ]
            if show_think_columns:
                test_parts.append(format_metric("test_nothink", include_target=False))
            test_values = "  ".join(test_parts)
            total_prompts = len(PROMPT_GOALS)
            remaining_prompts = prompt_tracker.remaining() if prompt_tracker else total_prompts
            solved_prompts = total_prompts - remaining_prompts
            line = (
                color_text(f"{total_steps}", Colors.CYAN)
                + " | "
                + color_text(train_values, Colors.GREEN)
                + " | "
                + color_text(test_values, Colors.MAGENTA)
                + " | "
                + color_text("sample: ", Colors.YELLOW)
                + colored_sample
            )
            print(line)
            eval_now = time.time()
            cycle_wall_elapsed = max(0.0, eval_now - cycle_wall_start)
            total_wall_seconds = base_wall_seconds + cycle_wall_elapsed

            record = {
                "step": total_steps,
                "train_loss": float(split_metrics["train"]["ce"]),
                "train_target": float(split_metrics["train"]["learned"]),
                "train_loss_nogrce": float(split_metrics["train_nogrce"]["ce"]),
                "train_target_nogrce": float(split_metrics["train_nogrce"]["learned"]),
                "test_loss": float(split_metrics["test"]["ce"]),
                "test_target": float(split_metrics["test"]["learned"]),
                "test_loss_nogrce": float(split_metrics["test_nogrce"]["ce"]),
                "test_target_nogrce": float(split_metrics["test_nogrce"]["learned"]),
                "train_wall_seconds": float(total_wall_seconds),
                "unix_time": float(eval_now),
                "train_cursor": int(dataset.positions.get("train", 0)),
                "test_cursor": int(dataset.positions.get("test", 0)),
            }
            if "train_nothink" in split_metrics:
                record["train_loss_plain"] = float(split_metrics["train_nothink"]["ce"])
                record["test_loss_plain"] = float(split_metrics["test_nothink"]["ce"])
            history_updates.append(record)
    
    if reward_tracker is not None:
        reward_tracker.finalize()
    return total_steps, history_updates


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
) -> None:
    model.eval()
    base_len = prompt_tokens.size(1)
    with torch.no_grad():
        for idx in range(1, count + 1):
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
            )
            completion_text = color_tokens(
                tokenizer,
                completion_ids,
                [Colors.YELLOW, Colors.CYAN],
                think_token_id=tokenizer.think_id,
                undo_token_id=tokenizer.undo_id,
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
    with torch.no_grad():
        logits, _, activations = model.forward_autoreg(
            seq,
            capture_activations=True,
            think_token_id=think_token_id,
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
) -> str:
    parts: list[str] = []
    color_index = 0
    completion_index = 0
    for idx, tok in enumerate(tokens):
        if think_token_id is not None and tok == think_token_id:
            piece = THINK_SYMBOL
            color = Colors.WHITE
            parts.append(color_text(piece, color, bold=True))
            color_index += 1
            continue
        if undo_token_id is not None and tok == undo_token_id:
            piece = UNDO_SYMBOL
            color = Colors.BLUE
            parts.append(color_text(piece, color, bold=True))
            color_index += 1
            continue
        piece = tokenizer.tokenizer.decode([tok], clean_up_tokenization_spaces=False)
        piece = tidy(piece, replace_newline=replace_newline)
        if not piece:
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
        parts.append(color_text(piece, color, bold=use_bold))
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
        if suppressed_ids:
            modified = probs.clone()
            modified[:, suppressed_ids] = 0
            sums = modified.sum(dim=-1, keepdim=True)
            mask = sums.squeeze(-1) > 0
            if mask.any():
                probs[mask] = modified[mask] / sums[mask]
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx, prompt_len


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_char_arg(value: str) -> int:
    value = value.strip().lower()
    if value.endswith("k"):
        return int(float(value[:-1]) * 1_000)
    if value.endswith("m"):
        return int(float(value[:-1]) * 1_000_000)
    return int(value)

def parse_args() -> argparse.Namespace:
    defaults = MODEL_CONFIG_TEMPLATE
    raw_cli_args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="simplewiki",
        help="Dataset base name; expects data/<name>-train.txt.gz and ...-test.txt.gz.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        help="Directory containing <corpus>-train.txt.gz and <corpus>-test.txt.gz",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--steps", type=int, default=100, help="Training steps per cycle")
    parser.add_argument(
        "--cycles",
        type=int,
        default=100,
        help="Repeat the full training/eval/update cycle N times.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=defaults.block_size,
        help="Number of tokens per training sample",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of sequences per optimization step.",
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=defaults.n_layer,
        help="Number of transformer blocks (GPT-2 base uses 12).",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=defaults.n_head,
        help="Number of attention heads per block (GPT-2 base uses 12).",
    )
    parser.add_argument(
        "--n-embd",
        type=int,
        default=defaults.n_embd,
        help="Embedding/hidden dimension (GPT-2 base uses 768).",
    )
    parser.add_argument(
        "--n-grce",
        type=int,
        default=defaults.n_grce,
        help="Dimension of the recurrent GRCE context; use 0 to disable the channel.",
    )
    parser.add_argument(
        "--context-span",
        type=int,
        default=2,
        help="Detach GRCE context gradients every N positions (0 disables detaching).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=defaults.dropout,
        help="Dropout probability inside attention/FFN blocks.",
    )
    parser.add_argument(
        "--nogrce-interval",
        type=int,
        default=1,
        help="Apply GRCE-disable dropout every N steps (0 disables)",
    )
    parser.add_argument(
        "--reward-relu",
        type=float,
        default=0.0,
        help=(
            "Enable experimental ReLU reward updates with scale=10^{-value} (value<=0 disables)"
        ),
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="How often to run train/test evaluation steps.",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=5,
        help="How many mini-batches to average for evaluation losses.",
    )
    parser.add_argument(
        "--vocab-chars",
        type=str,
        default="16M",
        help="Optional limit on how many characters to feed into the tokenizer trainer.",
    )
    parser.add_argument(
        "--train-chars",
        type=str,
        default="16M",
        help="Optional limit on how many characters of the training file to use.",
    )
    parser.add_argument(
        "--test-chars",
        type=str,
        default="4M",
        help="Optional limit on how many characters of the test file to use.",
    )
    parser.add_argument(
        "--generate",
        type=int,
        default=10,
        help="Number of new tokens to sample after training",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the standard training loop (default action if none specified)",
    )
    parser.add_argument(
        "--report",
        type=int,
        default=0,
        help="If >0, skip training and generate this many completions",
    )
    parser.add_argument(
        "--test",
        type=int,
        help="Print block_size tokens from the test corpus starting at cursor N",
    )
    parser.add_argument(
        "--no-newlines",
        action="store_true",
        help="During sampling/reporting, avoid emitting newline tokens",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="During sampling/reporting, suppress thinking tokens entirely",
    )
    parser.add_argument(
        "--no-think-prompt",
        action="store_true",
        help="Do not insert thinking tokens inside the prompt during sampling/reporting",
    )
    parser.add_argument(
        "--think-hard",
        action="store_true",
        help="While processing the prompt, insert thinking tokens after every mispredicted token",
    )
    parser.add_argument(
        "--think",
        type=int,
        default=0,
        help=(
            "Enable think mode with up to N inserted thinking tokens per block; "
            "adds the <think> special token"
        ),
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help=(
            "Shortcut for --block-size 16 --batch-size 4 --n-layer 2 --n-head 2 "
            "--n-embd 64 --n-grce 16"
        ),
    )
    parser.add_argument(
        "--grce-layered",
        action="store_true",
        help="Use per-layer GRCE sampling MLPs (requires n_grce %% n_layer == 0)",
    )
    parser.add_argument(
        "--print-size",
        action="store_true",
        help="Print parameter breakdown for the configured model and exit",
    )
    parser.add_argument(
        "--prompt-cycle-prompts",
        type=int,
        default=10,
        help="How many unsatisfied prompts to test each cycle",
    )
    parser.add_argument(
        "--undo",
        type=int,
        default=0,
        help="Enable undo pairs with up to N random+undo sequences per block",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Append an extra _TAG suffix to the model name (can be repeated)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="ai will",
        help="Prompt used for generation",
    )
    parser.add_argument(
        "--tokenizer-vocab",
        type=int,
        default=defaults.vocab_size,
        help="Vocabulary size for the GPT-2 style byte-level BPE tokenizer.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model",
        help="Directory where checkpoints/logs/tokenizers are stored",
    )
    parser.add_argument(
        "--debug-interrupt",
        action="store_true",
        help="If set, re-raise KeyboardInterrupt with a full stack trace.",
    )
    parser.add_argument(
        "--no-ansi",
        action="store_true",
        help="Suppress the parallel .ansi log (which preserves ANSI colors)",
    )
    parser.add_argument(
        "--import-model",
        type=pathlib.Path,
        help="Initialize from another checkpoint when creating a new model",
    )
    parser.add_argument(
        "--trim-model",
        action="store_true",
        help="Allow importing into a smaller model by dropping overflow",
    )
    parser.add_argument(
        "--drop-layers",
        type=str,
        default="",
        help="Comma-separated layer numbers (1-indexed) to remove during import",
    )
    parser.add_argument(
        "--add-layers",
        type=str,
        default="",
        help="Comma-separated layer numbers (1-indexed) to insert during import",
    )
    args = parser.parse_args()
    if args.tiny:
        def flag_present(flag: str) -> bool:
            return any(arg == flag or arg.startswith(f"{flag}=") for arg in raw_cli_args)

        if not flag_present("--block-size"):
            args.block_size = 16
        if not flag_present("--batch-size"):
            args.batch_size = 4
        if not flag_present("--n-layer"):
            args.n_layer = 2
        if not flag_present("--n-head"):
            args.n_head = 2
        if not flag_present("--n-embd"):
            args.n_embd = 64
        if not flag_present("--n-grce"):
            args.n_grce = 16
    return args


def main() -> None:
    args = parse_args()
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
            "Prompt contains thinking tokens but --think is 0. Remove them or enable --think."
        )
    if args.undo == 0 and UNDO_TOKEN in args.prompt:
        raise ValueError(
            "Prompt contains undo tokens but --undo is 0. Remove them or enable --undo."
        )
    torch.manual_seed(42)
    random.seed(42)

    actions: list[str] = []
    if args.train:
        actions.append("train")
    if args.report > 0:
        actions.append("report")
    if args.test is not None:
        actions.append("test")
    if not actions:
        actions.append("train")
    if len(actions) > 1:
        raise ValueError("Specify only one of --train, --report, or --test")
    selected_action = actions[0]

    ansi_file = None
    try:
        orig_stdout, orig_stderr, log_file = sys.stdout, sys.stderr, None

        data_dir = pathlib.Path(args.data)
        train_path = data_dir / f"{args.corpus}-train.txt.gz"
        test_path = data_dir / f"{args.corpus}-test.txt.gz"
        train_limit = parse_char_arg(args.train_chars)
        test_limit = parse_char_arg(args.test_chars)
        vocab_limit = parse_char_arg(args.vocab_chars)
        tokenizer_limit = parse_char_arg(args.vocab_chars or args.train_chars)

        model_dir = pathlib.Path(args.model)
        model_dir.mkdir(parents=True, exist_ok=True)

        def limit_label(value: int) -> str:
            return str(value if value > 0 else "all")

        train_cache_path = (
            model_dir
            / f"{args.corpus}_tokens_train_{limit_label(train_limit)}_{args.tokenizer_vocab}.pt"
        )
        test_cache_path = (
            model_dir
            / f"{args.corpus}_tokens_test_{limit_label(test_limit)}_{args.tokenizer_vocab}.pt"
        )

        try:
            full_train_text = load_text_file(train_path)
        except FileNotFoundError:
            if not train_cache_path.exists():
                raise
            full_train_text = None

        try:
            full_test_text = load_text_file(test_path)
        except FileNotFoundError:
            if not test_cache_path.exists():
                raise
            full_test_text = None

        tokenizer_key = (
            f"{args.corpus}_vocab_{limit_label(tokenizer_limit)}_{args.tokenizer_vocab}"
        )
        tokenizer_path = model_dir / f"{tokenizer_key}.json"
        if not tokenizer_path.exists() and full_train_text is None:
            raise FileNotFoundError(
                f"Tokenizer cache {tokenizer_path} not found and training text is unavailable."
            )
        print(color_text(f"Tokenizer: {tokenizer_path}", Colors.BLUE))
        tok_wall_start = time.time()
        tok_cpu_start = time.process_time()
        vocab_source = ""
        if full_train_text is not None:
            vocab_source = full_train_text if vocab_limit == 0 else full_train_text[:vocab_limit]
        tokenizer = GPT2TokenizerWrapper(
            vocab_source,
            tokenizer_path,
            args.tokenizer_vocab,
        )

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

        train_tokens, train_text, train_bytes, train_inserts = load_or_prepare_tokens(
            "train",
            train_path,
            full_train_text,
            train_limit,
            train_cache_path,
            tokenizer,
            seed=1234,
        )

        test_tokens, test_text, test_bytes, test_inserts = load_or_prepare_tokens(
            "test",
            test_path,
            full_test_text,
            test_limit,
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

        tok_summary = (
            f"[tokenizer] wall={time.time()-tok_wall_start:.2f}s cpu={time.process_time()-tok_cpu_start:.2f}s\n"
        )
        print(tok_summary)

        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            n_grce=args.n_grce,
            dropout=args.dropout,
            context_span=max(0, args.context_span),
            context_dropout=1,
            grce_layered=args.grce_layered,
        )
        if args.print_size:
            describe_model_size(config)
            return
        model_tag = build_model_tag(config)
        if args.think > 0:
            model_tag += f"_think{args.think}"
        if args.undo > 0:
            model_tag += f"_undo{args.undo}"
        for extra_tag in args.tag:
            cleaned = re.sub(r"[^0-9A-Za-z]+", "", extra_tag)
            if cleaned:
                model_tag += f"_{cleaned}"
        prefix = f"{args.corpus}_model_"
        model_path = model_dir / f"{prefix}{model_tag}.pt"
        log_path = model_dir / f"{prefix}{model_tag}.log"
        print(color_text(f"Model: {model_path}", Colors.CYAN))
        print(color_text(f"Logfile: {log_path}", Colors.BLUE))
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
            if "Torch not compiled with CUDA" in message:
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
                print(color_text("Torch not compiled with CUDA enabled; switching to CPU", Colors.RED, bold=True))
                device = torch.device("cpu")
                model = GRCEGPT(config).to(device)
            else:
                raise
        total_steps = 0
        loss_history: List[Dict[str, float]] = []
        total_train_wall = 0.0
        if model_path.exists():
            if args.import_model:
                raise ValueError(
                    "--import-model can only be used when no existing checkpoint is present"
                )
            payload = torch.load(
                model_path,
                map_location=device,
                weights_only=False,  # checkpoints also store dataset offsets/counters
            )
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
                if prompt_tracker.remaining() < len(PROMPT_GOALS):
                    satisfied = [
                        idx
                        for idx, done in enumerate(prompt_tracker.completed)
                        if done
                    ]
                    if satisfied:
                        total_prompts = len(PROMPT_GOALS)
                        lines = []
                        for idx in satisfied:
                            text, expected = PROMPT_GOALS[idx]
                            lines.append(
                                color_text(
                                    f"#{idx + 1}: '{text}' -> '{expected}'",
                                    Colors.GREEN,
                                )
                            )
                        print(
                            "\n"
                            + color_text(
                                f"Satisfied prompts ({len(satisfied)}/{total_prompts}):",
                                Colors.GREEN,
                                bold=True,
                            )
                            + "\n"
                            + "\n".join(lines)
                            + "\n"
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

        if selected_action == "report":
            run_report_mode(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                sample_len=args.generate,
                count=args.report,
                device=device,
                suppress_newlines=args.no_newlines,
                newline_token_id=newline_token_id,
                think_settings=think_settings,
                suppress_think=args.no_think,
                suppress_think_prompt=args.no_think_prompt,
                think_hard=args.think_hard,
            )
            return

        if selected_action == "test":
            run_test_slice(
                dataset=dataset,
                tokenizer=tokenizer,
                model=model,
                block_size=args.block_size,
                start_pos=args.test,
                think_settings=think_settings,
            )
            return

        reward_scale = 0.0
        if args.reward_relu > 0:
            reward_scale = 10 ** (-float(args.reward_relu))
        for cycle in range(1, args.cycles + 1):
            cycle_wall = time.time()
            cycle_cpu = time.process_time()
            tags = ["GPT"]
            plus_tags: list[str] = []
            minus_tags: list[str] = []
            if args.n_grce > 0:
                plus_tags.append("+GRCE")
            else:
                minus_tags.append(" wo/GRCE")
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
            print(color_text(f"\nModel: {model_path}", Colors.CYAN))
            print(
                color_text(
                    f"[{label}] Training Cycle {cycle}/{args.cycles}. "
                    f"Total training so far: {total_steps} steps, {hours:.2f} hours ({days:.2f} days)",
                    Colors.BLUE,
                )
            )
            cycle_prompt_indices: List[int] | None = None
            if prompt_tracker is not None and args.prompt_cycle_prompts > 0:
                cycle_prompt_indices = prompt_tracker.pending_indices(args.prompt_cycle_prompts)
                if cycle_prompt_indices:
                    preview_lines = []
                    for idx in cycle_prompt_indices:
                        text, expected = PROMPT_GOALS[idx]
                        preview_lines.append(f"#{idx + 1}: '{text}' -> '{expected}'")
            train_chars_cycle = (args.block_size + 1) * args.batch_size * args.steps
            test_chars_cycle = (
                (args.block_size + 1)
                * args.batch_size
                * max(1, args.eval_iters * count_eval_calls(args.steps, args.eval_interval))
            )
            dataset.prepare_cycle("train", train_chars_cycle)
            dataset.prepare_cycle("test", test_chars_cycle)

            total_steps, updates = train_model(
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
                suppress_think_output=args.no_think,
                suppress_think_prompt=args.no_think_prompt,
                think_hard=args.think_hard,
                undo_settings=undo_settings,
                prompt_tracker=prompt_tracker,
                reward_relu=reward_scale,
                nogrce_interval=args.nogrce_interval,
                cycle_wall_start=cycle_wall,
                base_wall_seconds=total_train_wall,
                cycle_prompt_indices=cycle_prompt_indices,
            )
            loss_history.extend(updates)

            train_wall = time.time() - cycle_wall
            train_cpu = time.process_time() - cycle_cpu
            total_train_wall += train_wall

            save_wall_start = time.time()
            save_cpu_start = time.process_time()
            torch.save(
                {
                    "model": model.state_dict(),
                    "dataset": dataset.state_dict(),
                    "total_steps": total_steps,
                    "loss_history": loss_history,
                    "config": asdict(config),
                    "train_wall_seconds": total_train_wall,
                    "prompt_state": prompt_tracker.serialize() if prompt_tracker else None,
                },
                model_path,
            )
            if log_file is not None:
                log_file.flush()
            if ansi_file is not None:
                ansi_file.flush()
            save_wall = time.time() - save_wall_start
            save_cpu = time.process_time() - save_cpu_start
            print(
                color_text(
                    f"[cycle {cycle}] total steps: {total_steps}; "
                    f"time spent: wall={train_wall:.2f}s cpu={train_cpu:.2f}s; "
                    f"writing model: wall={save_wall:.2f}s cpu={save_cpu:.2f}s",
                    Colors.CYAN,
                )
            )

    except KeyboardInterrupt:
        if args.debug_interrupt:
            raise
        print(color_text("Interrupted by user; exiting cleanly.", Colors.MAGENTA))

    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        if log_file is not None:
            log_file.close()
        if ansi_file is not None:
            ansi_file.close()


if __name__ == "__main__":
    main()
