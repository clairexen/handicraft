"""GRCE proof-of-concept. Most of it is written by ChatGPT/Codex. I told
it to base it loosely the picoGPT.

This script keeps the picoGPT spirit of being small and hackable while
adding the Gradient-limited Recurrent Context Encoding (GRCE) channel described in
``grce.md``. It trains a tiny character-level Transformer on the bundled
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
from dataclasses import dataclass, field
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

DISSONANCE_TOKEN = "<|?!|>"
DISSONANCE_RATE = 0.01
FANCY_SPACE = "\u2423"  # Open Box symbol for visible spaces
FANCY_ENTER = "\u23CE"  # Return symbol for visible newlines
THINK_TOKEN = "<think>"
THINK_SYMBOL = "\u2754"  # white question mark
UNDO_TOKEN = "<undo>"
UNDO_SYMBOL = "\u21A9"  # leftwards arrow with hook
ASCII_LETTERS = set(string.ascii_letters)


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
        use_special: bool,
        use_think: bool,
        use_undo: bool,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_path
        self.use_special = use_special
        self.use_think = use_think
        self.use_undo = use_undo
        self.extra_special_tokens: list[str] = []
        if self.use_special:
            self.extra_special_tokens.append(DISSONANCE_TOKEN)
        if self.use_think:
            self.extra_special_tokens.append(THINK_TOKEN)
        if self.use_undo:
            self.extra_special_tokens.append(UNDO_TOKEN)
        self.tokenizer = self._load_or_train(
            train_text,
            cache_path,
            vocab_size,
            self.extra_special_tokens,
        )
        self.vocab_size = len(self.tokenizer)
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.dissonance_id = None
        if self.use_special:
            self.dissonance_id = self.tokenizer.convert_tokens_to_ids(DISSONANCE_TOKEN)
            if self.dissonance_id is None:
                raise ValueError("Failed to add dissonance token to tokenizer vocabulary")
        self.think_id = None
        if self.use_think:
            self.think_id = self.tokenizer.convert_tokens_to_ids(THINK_TOKEN)
            if self.think_id is None:
                raise ValueError("Failed to add think token to tokenizer vocabulary")
        self.undo_id = None
        if self.use_undo:
            self.undo_id = self.tokenizer.convert_tokens_to_ids(UNDO_TOKEN)
            if self.undo_id is None:
                raise ValueError("Failed to add undo token to tokenizer vocabulary")
        self.non_special_ids = [
            tok_id for tok_id in range(self.vocab_size) if tok_id not in self.special_ids
        ]
        if self.use_special and not self.non_special_ids:
            raise ValueError("Tokenizer has no non-special tokens for dissonance markers")

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
    fraction: float = 1.0

    def __post_init__(self) -> None:
        frac = 0.0 if self.fraction is None else float(self.fraction)
        self.fraction = max(0.0, min(1.0, frac))

    @property
    def enabled(self) -> bool:
        return self.max_steps > 0 and self.token_id is not None


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


def insert_dissonance_markers(
    tokens: torch.Tensor,
    non_special_ids: List[int],
    marker_id: int,
    rate: float,
    rng: random.Random,
) -> Tuple[torch.Tensor, int]:
    if not (0.0 < rate < 1.0):
        return tokens.clone(), 0
    base = tokens.tolist()
    augmented: List[int] = []
    inserts = 0
    for tok in base:
        augmented.append(int(tok))
        if rng.random() < rate:
            filler = rng.choice(non_special_ids)
            augmented.append(filler)
            augmented.append(marker_id)
            inserts += 1
    return torch.tensor(augmented, dtype=torch.long), inserts


def augment_training_batch(
    model: GRCEGPT,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    think: ThinkSettings | None,
    undo: UndoSettings | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    think_enabled = think is not None and think.enabled
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
    think_scores = None
    prev_mode = model.training
    active_rows: set[int] = set()
    if think_enabled:
        think_labels = torch.full_like(inputs, -1)
        think_slot_mask = torch.zeros_like(inputs, dtype=torch.bool, device=device)
        allowed = int(round(think.fraction * B))
        allowed = max(0, min(B, allowed))
        if allowed == 0 and think.fraction > 0.0:
            allowed = 1
        if allowed > 0:
            row_indices = list(range(B))
            random.shuffle(row_indices)
            active_rows = set(row_indices[:allowed])
        model.eval()
        with torch.no_grad():
            logits, _, _ = model.forward_autoreg(inputs)
            think_scores = logits[..., think.token_id]
        if prev_mode:
            model.train()
    for row in range(B):
        row_think_active = think_enabled and row in active_rows
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
            if not undo.fill_choices:
                raise ValueError("Undo mode requires non-empty filler token choices")
            for _ in range(undo_pairs):
                filler = int(random.choice(undo.fill_choices))
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
    use_special: bool,
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
    inserts = 0
    if use_special:
        tokens, inserts = insert_dissonance_markers(
            tokens,
            tokenizer.non_special_ids,
            tokenizer.dissonance_id,
            DISSONANCE_RATE,
            random.Random(seed),
        )
    bytes_count = len(trimmed_text.encode("utf-8"))
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
    context_dropout: int = 0  # Every N positions drop GRCE connection (0 disables).


MODEL_CONFIG_TEMPLATE = ModelConfig()


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
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        pre_ff = self.ln2(x)
        ff_out = self.ff(pre_ff)
        x = x + ff_out
        return x, pre_ff, ff_out


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
        self, idx: torch.Tensor, block_biases: List[torch.Tensor] | None = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, T = idx.shape
        device = idx.device
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = self.drop(tok + pos)
        block_inputs: List[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            if block_biases is not None:
                x = x + block_biases[layer_idx]
            block_inputs.append(x[:, -1, :])
            x, _, _ = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, block_inputs


class GRCEContextChannel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.disabled = config.n_grce <= 0
        self.config = config
        self.context_span = max(0, int(config.context_span))
        self.context_dim = config.n_grce
        self.context_dropout = max(0, int(config.context_dropout))
        if not self.disabled:
            mid = 4 * config.n_grce
            self.pre_norms = nn.ModuleList(
                nn.LayerNorm(config.n_embd) for _ in range(config.n_layer)
            )
            self.context_sampler = nn.ModuleList(
                nn.Linear(config.n_embd, config.n_grce)
                for _ in range(config.n_layer)
            )
            self.context_mlp = nn.Sequential(
                nn.Linear(config.n_grce, mid),
                nn.ReLU(),
                nn.LayerNorm(mid),
                nn.Linear(mid, config.n_grce),
            )
            self.context_norm = nn.LayerNorm(config.n_grce)
            self.context_bias_gen = nn.ModuleList(
                nn.Linear(config.n_grce, config.n_embd)
                for _ in range(config.n_layer)
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
        stop_grad: bool,
    ) -> torch.Tensor:
        if self.disabled:
            raise RuntimeError("Context channel disabled; update should not be called.")
        pieces = [inp.detach() if stop_grad else inp for inp in block_inputs]
        messages = []
        for ln, sampler, part in zip(self.pre_norms, self.context_sampler, pieces):
            messages.append(sampler(ln(part)))
        fused = torch.stack(messages, dim=0).sum(dim=0)
        context = self.context_mlp(fused)
        context = self.context_norm(context)
        return context


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
        drop_mask: set[int] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        B, T = idx.shape
        device = idx.device
        use_context = not self.context.disabled and not disable_context
        context = None
        if use_context:
            context_dim = self.context.context_dim
            context = torch.zeros(B, context_dim, device=device)
        logits_steps = []
        for t in range(T):
            prefix = idx[:, : t + 1]
            tok_last = self.core.tok_emb(prefix[:, -1])
            pos_ids = torch.full((B,), t, device=device, dtype=torch.long)
            pos_emb = self.core.pos_emb(pos_ids)
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
            logits, block_inputs = self.core(prefix, block_biases=block_biases)
            if use_context and context is not None:
                span = self.context.context_span
                if span <= 0:
                    stop_grad = False
                elif span == 1:
                    stop_grad = True
                else:
                    stop_grad = (t % span == 0)
                drop_ratio = self.context.context_dropout
                use_grce = True
                if drop_ratio == 1 and drop_mask is not None and self.training:
                    slot = t % self.config.block_size if self.config.block_size > 0 else 0
                    use_grce = slot not in drop_mask
                elif drop_ratio > 1 and self.training:
                    use_grce = (random.randrange(drop_ratio) != 0)
                if use_grce:
                    context = self.context.update(block_inputs, stop_grad=stop_grad)
            logits_steps.append(logits[:, -1:, :])
        logits = torch.cat(logits_steps, dim=1)
        return logits, context, None


def build_model_tag(config: ModelConfig) -> str:
    tag = (
        f"v{config.vocab_size}_bs{config.block_size}_emb{config.n_embd}_"
        f"layers{config.n_layer}_heads{config.n_head}_ctx{config.n_grce}"
    )
    if config.n_grce > 0:
        tag += f"_span{config.context_span}"
        if config.context_dropout > 0:
            tag += f"_drop{config.context_dropout}"
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
        logits, _, _ = model.forward_autoreg(prefix)
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
    drop_positions: list[set[int]] | None,
    think_settings: ThinkSettings | None,
    suppress_think_output: bool,
    suppress_think_prompt: bool,
    think_hard: bool,
    undo_settings: UndoSettings | None,
) -> Tuple[int, List[Dict[str, float]]]:
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    total_steps = start_step
    history_updates: List[Dict[str, float]] = []
    printed_header = False
    think_enabled = think_settings is not None and think_settings.enabled
    undo_enabled = undo_settings is not None and undo_settings.enabled
    show_think_columns = think_enabled
    show_learned_headers = think_enabled or undo_enabled
    for step in range(1, steps + 1):
        xb, yb = dataset.get_batch("train", block_size, batch_size, device)
        xb, yb, random_mask, think_labels, think_slot_mask = augment_training_batch(
            model,
            xb,
            yb,
            think_settings,
            undo_settings,
        )
        drop_mask = None
        if drop_positions is not None and 0 <= step - 1 < len(drop_positions):
            drop_mask = drop_positions[step - 1]
        logits, _, _ = model.forward_autoreg(
            xb,
            yb,
            drop_mask=drop_mask,
        )
        logits = apply_think_slot_mask(logits, think_slot_mask, think_settings)
        logits_flat = logits.view(-1, logits.size(-1))
        loss_targets = build_loss_targets(yb, think_settings, random_mask)
        main_loss = F.cross_entropy(
            logits_flat,
            loss_targets.view(-1),
            ignore_index=LOSS_IGNORE_INDEX,
        )
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
                            undo_settings=undo_settings,
                            batches=cached_batches[split],
                        )
                        split_metrics[f"{split}_nothink"] = {
                            "ce": float(ce_loss),
                            "learned": float(learned_loss),
                        }
                sample_tokens, prompt_len = generate(
                    model,
                    sample_prompt.clone(),
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

            prefix_text = color_tokens(
                tokenizer,
                prompt_ids,
                [Colors.CYAN, Colors.GREEN],
                bold=False,
                think_token_id=tokenizer.think_id,
                undo_token_id=tokenizer.undo_id,
            )
            completion_text = color_tokens(
                tokenizer,
                completion_ids,
                [Colors.YELLOW, Colors.MAGENTA],
                think_token_id=tokenizer.think_id,
                undo_token_id=tokenizer.undo_id,
            )
            colored_sample = prefix_text + completion_text
            if not printed_header:
                if show_learned_headers:
                    train_header = "train loss (learned)  nogrce (learned)"
                    test_header = "test loss (learned)  nogrce (learned)"
                else:
                    train_header = "train loss  nogrce"
                    test_header = "test loss  nogrce"
                if show_think_columns:
                    train_header += "  nothink"
                    test_header += "  nothink"
                header_line = (
                    color_text("step", Colors.CYAN)
                    + " | "
                    + color_text(train_header, Colors.GREEN)
                    + " | "
                    + color_text(test_header, Colors.MAGENTA)
                    + " | sample"
                )
                print(header_line)
                printed_header = True

            def format_metric(key: str) -> str:
                metric = split_metrics[key]
                ce_val = metric["ce"]
                learned_val = metric["learned"]
                if abs(learned_val - ce_val) < 1e-6:
                    return f"{ce_val:.2f}"
                return f"{ce_val:.2f} ({learned_val:.2f})"

            train_parts = [
                format_metric("train"),
                format_metric("train_nogrce"),
            ]
            if show_think_columns:
                train_parts.append(format_metric("train_nothink"))
            train_values = "  ".join(train_parts)

            test_parts = [
                format_metric("test"),
                format_metric("test_nogrce"),
            ]
            if show_think_columns:
                test_parts.append(format_metric("test_nothink"))
            test_values = "  ".join(test_parts)
            line = (
                color_text(f"{step:04d}", Colors.CYAN)
                + " | "
                + color_text(train_values, Colors.GREEN)
                + " | "
                + color_text(test_values, Colors.MAGENTA)
                + " | sample: "
                + colored_sample
            )
            print(line)
            record = {
                "step": total_steps,
                "train_loss": float(split_metrics["train"]["ce"]),
                "train_loss_learned": float(split_metrics["train"]["learned"]),
                "train_loss_nogrce": float(split_metrics["train_nogrce"]["ce"]),
                "train_loss_nogrce_learned": float(split_metrics["train_nogrce"]["learned"]),
                "test_loss": float(split_metrics["test"]["ce"]),
                "test_loss_learned": float(split_metrics["test"]["learned"]),
                "test_loss_nogrce": float(split_metrics["test_nogrce"]["ce"]),
                "test_loss_nogrce_learned": float(split_metrics["test_nogrce"]["learned"]),
            }
            if "train_nothink" in split_metrics:
                record["train_loss_nothink"] = float(split_metrics["train_nothink"]["ce"])
                record["train_loss_nothink_learned"] = float(
                    split_metrics["train_nothink"]["learned"]
                )
                record["test_loss_nothink"] = float(split_metrics["test_nothink"]["ce"])
                record["test_loss_nothink_learned"] = float(
                    split_metrics["test_nothink"]["learned"]
                )
            history_updates.append(record)
    
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
                [Colors.CYAN, Colors.GREEN],
                bold=False,
                think_token_id=tokenizer.think_id,
                undo_token_id=tokenizer.undo_id,
            )
            completion_text = color_tokens(
                tokenizer,
                completion_ids,
                [Colors.YELLOW, Colors.MAGENTA],
                think_token_id=tokenizer.think_id,
                undo_token_id=tokenizer.undo_id,
            )
            print(
                color_text(f"[report {idx:02d}]", Colors.CYAN)
                + " | sample: "
                + prefix_text
                + completion_text
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
) -> str:
    parts: list[str] = []
    color_index = 0
    for tok in tokens:
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
        color = colors[color_index % len(colors)]
        parts.append(color_text(piece, color, bold=bold))
        color_index += 1
    return "".join(parts)


def normalize_prompt(text: str) -> str:
    return text.replace(FANCY_SPACE, " ").replace(FANCY_ENTER, "\n")


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
    if not suppress_think and not suppress_think_prompt:
        if think_hard and think_settings is not None and think_settings.enabled:
            with torch.no_grad():
                logits, _, _ = model.forward_autoreg(idx)
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
        logits, _, _ = model.forward_autoreg(idx_cond)
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=str,
        default="simplewiki",
        help="Dataset base name; expects data/<name>-train.txt.gz and ...-test.txt.gz.",
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
        "--report-count",
        type=int,
        default=0,
        help="If >0, skip training and generate this many completions",
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
        "--grce-dropout",
        type=int,
        default=0,
        help="Drop GRCE connections every N positions (0 disables)",
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
        "--think-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of sequences per batch (0-1) that participate in "
            "thinking; 1.0 restores the previous behavior"
        ),
    )
    parser.add_argument(
        "--undo",
        type=int,
        default=0,
        help="Enable undo pairs with up to N random+undo sequences per block",
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
        "--special",
        action="store_true",
        help="Enable dissonance special tokens and insert markers into the dataset.",
    )
    parser.add_argument(
        "--debug-interrupt",
        action="store_true",
        help="If set, re-raise KeyboardInterrupt with a full stack trace.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.prompt = normalize_prompt(args.prompt)
    if args.think > 0:
        args.prompt = args.prompt.replace(THINK_SYMBOL, THINK_TOKEN)
    if args.undo > 0:
        args.prompt = args.prompt.replace(UNDO_SYMBOL, UNDO_TOKEN)
    torch.manual_seed(42)
    random.seed(42)

    try:
        orig_stdout, orig_stderr, log_file = sys.stdout, sys.stderr, None

        train_path = pathlib.Path("data") / f"{args.data}-train.txt.gz"
        test_path = pathlib.Path("data") / f"{args.data}-test.txt.gz"
        train_limit = parse_char_arg(args.train_chars)
        test_limit = parse_char_arg(args.test_chars)
        vocab_limit = parse_char_arg(args.vocab_chars)
        tokenizer_limit = parse_char_arg(args.vocab_chars or args.train_chars)

        model_dir = pathlib.Path("model")
        model_dir.mkdir(parents=True, exist_ok=True)

        def limit_label(value: int) -> str:
            return str(value if value > 0 else "all")

        special_tag = "special" if args.special else "plain"
        think_tag = "think" if args.think > 0 else "nothink"
        undo_tag = "undo" if args.undo > 0 else "noundo"
        train_cache_path = (
            model_dir
            / f"{args.data}_tokens_train_{limit_label(train_limit)}_{args.tokenizer_vocab}_{special_tag}_{think_tag}_{undo_tag}.pt"
        )
        test_cache_path = (
            model_dir
            / f"{args.data}_tokens_test_{limit_label(test_limit)}_{args.tokenizer_vocab}_{special_tag}_{think_tag}_{undo_tag}.pt"
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
            f"{args.data}_vocab_{limit_label(tokenizer_limit)}_{args.tokenizer_vocab}_{special_tag}_{think_tag}_{undo_tag}"
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
            args.special,
            args.think > 0,
            args.undo > 0,
        )

        newline_token_id = None
        newline_tokens = tokenizer.tokenizer.encode("\n", add_special_tokens=False)
        if newline_tokens:
            newline_token_id = newline_tokens[0]

        think_settings = ThinkSettings(
            max_steps=args.think,
            token_id=tokenizer.think_id,
            fraction=args.think_fraction,
        )
        undo_settings = UndoSettings(
            max_pairs=args.undo,
            token_id=tokenizer.undo_id,
            fill_choices=tokenizer.non_special_ids,
        )

        train_tokens, train_text, train_bytes, train_inserts = load_or_prepare_tokens(
            "train",
            train_path,
            full_train_text,
            train_limit,
            train_cache_path,
            tokenizer,
            seed=1234,
            use_special=args.special,
        )

        test_tokens, test_text, test_bytes, test_inserts = load_or_prepare_tokens(
            "test",
            test_path,
            full_test_text,
            test_limit,
            test_cache_path,
            tokenizer,
            seed=5678,
            use_special=args.special,
        )
        if args.special:
            print(
                color_text(
                    (
                        f"Dissonance injections (train/test): "
                        f"{train_inserts}/{test_inserts} sequences"
                    ),
                    Colors.CYAN,
                )
            )
        else:
            print(
                color_text(
                    "Special token insertions disabled (--special not set)",
                    Colors.GRAY,
                )
            )

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
            context_dropout=max(0, args.grce_dropout),
        )
        model_tag = build_model_tag(config)
        if args.think > 0:
            model_tag += f"_think{args.think}"
        if args.undo > 0:
            model_tag += f"_undo{args.undo}"
        prefix = f"{args.data}_model_"
        model_path = model_dir / f"{prefix}{model_tag}.pt"
        log_path = model_dir / f"{prefix}{model_tag}.log"
        print(color_text(f"Model: {model_path}", Colors.BLUE))
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

        sys.stdout = Tee((orig_stdout, False), (log_file, True))
        sys.stderr = Tee((orig_stderr, False), (log_file, True))

        device = torch.device(args.device)
        try:
            prompt_tokens = tokenizer.encode(args.prompt)
        except KeyError as exc:  # pragma: no cover - user misconfiguration
            raise ValueError(
                "Prompt contains characters outside the tokenizer vocabulary. "
                "Choose a simpler prompt or extend the dataset."
            ) from exc
        prompt_tokens = prompt_tokens.unsqueeze(0).to(device)

        model = GRCEGPT(config).to(device)
        total_steps = 0
        loss_history: List[Dict[str, float]] = []
        if model_path.exists():
            payload = torch.load(
                model_path,
                map_location=device,
                weights_only=False,  # checkpoints also store dataset offsets/counters
            )
            try:
                if isinstance(payload, dict) and "model" in payload:
                    model.load_state_dict(payload["model"])
                    if "dataset" in payload:
                        dataset.load_state(payload["dataset"])
                    total_steps = int(payload.get("total_steps", 0))
                    loss_history = list(payload.get("loss_history", []))
                else:
                    model.load_state_dict(payload)
                print(color_text(f"Loaded existing model from {model_path}", Colors.YELLOW))
                print(color_text(f"Total steps so far: {total_steps}", Colors.YELLOW))
            except RuntimeError as err:
                print(
                    color_text(
                        "Checkpoint load failed (shape mismatch); starting fresh.",
                        Colors.MAGENTA,
                    )
                )
                print(color_text(str(err), Colors.GRAY))

        if args.report_count > 0:
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
                suppress_think=args.no_think,
                suppress_think_prompt=args.no_think_prompt,
                think_hard=args.think_hard,
            )
            return

        for cycle in range(1, args.cycles + 1):
            drop_positions = None
            if args.grce_dropout == 1 and args.n_grce > 0:
                drop_positions = []
                for _ in range(args.steps):
                    span = max(args.block_size - 1, 1)
                    slots = list(range(span))
                    picks = random.randrange(span + 1)
                    drop_positions.append(set(random.sample(slots, picks)))

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
            print(color_text(f"\n[{label}] Training Cycle {cycle}/{args.cycles} ...", Colors.BLUE))
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
                drop_positions=drop_positions,
                think_settings=think_settings,
                suppress_think_output=args.no_think,
                suppress_think_prompt=args.no_think_prompt,
                think_hard=args.think_hard,
                undo_settings=undo_settings,
            )
            loss_history.extend(updates)
            print(color_text(f"Total steps so far: {total_steps}", Colors.YELLOW))

            torch.save(
                {
                    "model": model.state_dict(),
                    "dataset": dataset.state_dict(),
                    "total_steps": total_steps,
                    "loss_history": loss_history,
                },
                model_path,
            )
            if log_file is not None:
                log_file.flush()
            print(color_text(f"Saved model to {model_path}", Colors.GREEN))
            cycle_elapsed_wall = time.time() - cycle_wall
            cycle_elapsed_cpu = time.process_time() - cycle_cpu
            print(
                color_text(
                    f"[cycle {cycle}] wall={cycle_elapsed_wall:.2f}s cpu={cycle_elapsed_cpu:.2f}s",
                    Colors.GRAY,
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


if __name__ == "__main__":
    main()
