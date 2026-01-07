"""GRCE proof-of-concept based on the picoGPT Shakespeare demo.

This script keeps the picoGPT spirit of being small and hackable while
adding the Gradient-less Recurrent Context Encoding (GRCE) channel described in
``grce.md``. It trains a tiny character-level Transformer on the bundled
Simple English Wikipedia split and shows how the recurrent context vector can
be integrated with a stop-gradient constraint across time.
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


# -----------------------------------------------------------------------------
# Data utilities (borrow the spirit of picoGPT's Shakespeare example)
# -----------------------------------------------------------------------------


def load_text_file(path: pathlib.Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Provide a text file path.")
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
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_path
        self.tokenizer = self._load_or_train(train_text, cache_path, vocab_size)
        self.vocab_size = len(self.tokenizer)
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.dissonance_id = self.tokenizer.convert_tokens_to_ids(DISSONANCE_TOKEN)
        if self.dissonance_id is None:
            raise ValueError("Failed to add dissonance token to tokenizer vocabulary")
        self.non_special_ids = [
            tok_id for tok_id in range(self.vocab_size) if tok_id not in self.special_ids
        ]
        if not self.non_special_ids:
            raise ValueError("Tokenizer has no non-special tokens for dissonance markers")

    def _load_or_train(
        self, train_text: str, cache_path: pathlib.Path, vocab_size: int
    ) -> GPT2TokenizerFast:
        if cache_path.exists():
            return self._configure_special_tokens(
                GPT2TokenizerFast(tokenizer_file=str(cache_path))
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
        tokenizer.train_from_iterator([train_text], trainer=trainer)
        tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
        tokenizer.add_special_tokens([DISSONANCE_TOKEN])
        tokenizer.save(str(cache_path))
        tk = GPT2TokenizerFast(tokenizer_file=str(cache_path))
        return self._configure_special_tokens(tk)

    def _configure_special_tokens(self, tk: GPT2TokenizerFast) -> GPT2TokenizerFast:
        # Suggested GPT-2 style special tokens (BOS/EOS/UNK/PAD) are omitted for now.
        tk.add_special_tokens({"additional_special_tokens": [DISSONANCE_TOKEN]})
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
    train_text: str
    test_text: str
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
        segments = self._byte_segments(split, parts_text)
        self._log_segments(split, segments)
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
            texts.append(text[start : start + first_take])
        if second_take:
            parts.append(tokens[:second_take])
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

    def _log_segments(self, split: str, segments: list[tuple[int, int]]) -> None:
        if not segments:
            return
        label = "train" if split == "train" else "test"
        color = Colors.MAGENTA if split == "train" else Colors.YELLOW
        path = self.train_path if split == "train" else self.test_path
        seg_text = " + ".join(f"[{s},{e})" for s, e in segments)
        print(color_text(f"[{label}:{path.name}] bytes {seg_text}", color))


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


# -----------------------------------------------------------------------------
# Model components (picoGPT-style Transformer blocks + GRCE channel)
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    vocab_size: int = 2000  # GPT-2 base supports ~50k merges; we stay small for the PoC.
    block_size: int = 64    # GPT-2 base uses 1024 tokens.
    n_layer: int = 8        # GPT-2 base uses 12 layers.
    n_head: int = 8         # GPT-2 base uses 12 attention heads.
    n_embd: int = 192       # GPT-2 base uses 768 embedding dims.
    n_grce: int = 96        # GRCE context dims.
    dropout: float = 0.05


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
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        B, T = idx.shape
        device = idx.device
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = self.drop(tok + pos)
        ff_outputs: List[torch.Tensor] = []
        first_block_input: torch.Tensor | None = None
        for layer_idx, block in enumerate(self.blocks):
            if block_biases is not None:
                x = x + block_biases[layer_idx]
            if first_block_input is None:
                first_block_input = x
            x, _, ff_out = block(x)
            ff_outputs.append(ff_out)
        x = self.ln_f(x)
        logits = self.head(x)
        if first_block_input is None:
            first_block_input = torch.zeros_like(x)  # pragma: no cover
        return logits, ff_outputs, first_block_input


class GRCEContextChannel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.disabled = config.n_grce <= 0
        self.config = config
        if not self.disabled:
            concat_dim = (config.n_layer + 1) * config.n_embd
            self.part_norms = nn.ModuleList(
                nn.LayerNorm(config.n_embd) for _ in range(config.n_layer + 1)
            )
            self.stack_drop = nn.Dropout(config.dropout)
            self.writer = nn.Sequential(
                nn.Linear(concat_dim, 4 * config.n_grce),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(4 * config.n_grce, 4 * config.n_grce),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(4 * config.n_grce, config.n_grce),
                nn.LayerNorm(config.n_grce),
            )
            self.bias_generators = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(config.n_grce, 4 * config.n_grce),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(4 * config.n_grce, max(config.n_embd, config.n_grce)),
                    nn.Linear(max(config.n_embd, config.n_grce), 4 * config.n_embd),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(4 * config.n_embd, config.n_embd),
                )
                for _ in range(config.n_layer)
            )

    def project(self, context: torch.Tensor) -> List[torch.Tensor]:
        if self.disabled:
            raise RuntimeError("Context channel disabled; project should not be called.")
        return [gen(context) for gen in self.bias_generators]

    def update(
        self,
        block_input: torch.Tensor,
        ffn_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        if self.disabled:
            raise RuntimeError("Context channel disabled; update should not be called.")
        pieces = [block_input] + ffn_outputs
        normed = [norm(part) for norm, part in zip(self.part_norms, pieces)]
        fused = self.stack_drop(torch.cat(normed, dim=-1)).detach()
        return self.writer(fused)


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
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        B, T = idx.shape
        device = idx.device
        context = None
        if not self.context.disabled:
            context = torch.zeros(B, self.config.n_grce, device=device)
        logits_steps = []
        for t in range(T):
            prefix = idx[:, : t + 1]
            tok_last = self.core.tok_emb(prefix[:, -1])
            pos_ids = torch.full((B,), t, device=device, dtype=torch.long)
            pos_emb = self.core.pos_emb(pos_ids)
            token_input = tok_last + pos_emb
            block_biases = None
            if not self.context.disabled and context is not None:
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
            logits, ff_outputs, block0_inputs = self.core(
                prefix, block_biases=block_biases
            )
            curr_ff = [ff[:, -1, :] for ff in ff_outputs]
            if not self.context.disabled and context is not None:
                block0_last = block0_inputs[:, -1, :]
                context = self.context.update(block0_last, curr_ff)
            logits_steps.append(logits[:, -1:, :])
        logits = torch.cat(logits_steps, dim=1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1),
                targets.view(B * T),
            )
        return logits, context, loss


def build_model_tag(config: ModelConfig) -> str:
    return (
        f"v{config.vocab_size}_bs{config.block_size}_emb{config.n_embd}_"
        f"ctx{config.n_grce}_layers{config.n_layer}_heads{config.n_head}"
    )


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
) -> float:
    losses = []
    for _ in range(iters):
        xb, yb = dataset.get_batch(split, block_size, batch_size, device)
        _, _, loss = model.forward_autoreg(xb, yb)
        losses.append(loss.item())
    return sum(losses) / len(losses)


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
    prompt_text: str,
) -> Tuple[int, List[Dict[str, float]]]:
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    total_steps = start_step
    history_updates: List[Dict[str, float]] = []
    for step in range(1, steps + 1):
        xb, yb = dataset.get_batch("train", block_size, batch_size, device)
        logits, _, loss = model.forward_autoreg(xb, yb)
        if loss is None:
            raise RuntimeError("Loss should not be None during training")
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_steps += 1

        if step == 1 or step % eval_interval == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                split_losses = {
                    split: evaluate_split(
                        model, dataset, device, block_size, batch_size, split, eval_iters
                    )
                    for split in ("train", "test")
                }
                sample_tokens = generate(
                    model,
                    sample_prompt.clone(),
                    sample_chars,
                )
            model.train()
            sample_text = tokenizer.decode(sample_tokens[0].cpu()).replace("\n", " ")
            prefix = prompt_text.replace("\n", " ")
            if not sample_text.startswith(prefix):
                prefix = sample_text[: len(prefix)]
            completion = sample_text[len(prefix) :]
            colored_sample = prefix + color_text(completion, Colors.WHITE, bold=True)
            loss_text = (
                color_text(f"train loss {split_losses['train']:.3f}", Colors.GREEN)
                + " | "
                + color_text(f"test loss {split_losses['test']:.3f}", Colors.MAGENTA)
            )
            print(
                color_text(f"step {step:04d}", Colors.CYAN)
                + " | "
                + loss_text
                + " | sample: "
                + colored_sample
            )
            history_updates.append(
                {
                    "step": total_steps,
                    "train_loss": float(split_losses["train"]),
                    "test_loss": float(split_losses["test"]),
                }
            )
    
    return total_steps, history_updates


def count_eval_calls(steps: int, eval_interval: int) -> int:
    evals = 0
    for step in range(1, steps + 1):
        if step == 1 or step == steps or (eval_interval > 0 and step % eval_interval == 0):
            evals += 1
    return max(1, evals)


@torch.no_grad()
def generate(
    model: GRCEGPT,
    idx: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    model.eval()
    for _ in range(steps):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _, _ = model.forward_autoreg(idx_cond)
        logits_last = logits[:, -1, :]
        probs = F.softmax(logits_last, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx


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
        "--train-path",
        type=pathlib.Path,
        default=pathlib.Path("data/simplewiki-train.asc"),
        help="Training corpus file (default: Simple English Wikipedia split).",
    )
    parser.add_argument(
        "--test-path",
        type=pathlib.Path,
        default=pathlib.Path("data/simplewiki-test.asc"),
        help="Held-out corpus file for regular testing.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--steps", type=int, default=10, help="Training steps per cycle")
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
        default=8,
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
        "--dropout",
        type=float,
        default=defaults.dropout,
        help="Dropout probability inside attention/FFN blocks.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
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
        default=15,
        help="Number of new tokens to sample after training",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="bigotry is",
        help="Prompt used for generation",
    )
    parser.add_argument(
        "--tokenizer-vocab",
        type=int,
        default=defaults.vocab_size,
        help="Vocabulary size for the GPT-2 style byte-level BPE tokenizer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    random.seed(42)

    train_text = load_text_file(args.train_path)
    test_text = load_text_file(args.test_path)
    train_limit = parse_char_arg(args.train_chars)
    test_limit = parse_char_arg(args.test_chars)
    vocab_limit = parse_char_arg(args.vocab_chars)
    if train_limit > 0:
        train_text = train_text[:train_limit]
    if test_limit > 0:
        test_text = test_text[:test_limit]
    if not train_text:
        raise ValueError("Training text is empty; provide a larger corpus or lower --train-chars")
    tokenizer_dir = pathlib.Path("model")
    tokenizer_limit = parse_char_arg(args.vocab_chars or args.train_chars)
    tokenizer_key = (
        f"{args.train_path.stem}_{tokenizer_limit or 'all'}_{args.tokenizer_vocab}"
    )
    tokenizer_path = tokenizer_dir / f"{tokenizer_key}.json"
    print(color_text(f"Tokenizer: {tokenizer_path}", Colors.BLUE))
    tok_wall_start = time.time()
    tok_cpu_start = time.process_time()
    vocab_source = train_text if vocab_limit == 0 else train_text[:vocab_limit]
    tokenizer = GPT2TokenizerWrapper(
        vocab_source,
        tokenizer_path,
        args.tokenizer_vocab,
    )

    print(color_text(f"Train Data: {args.train_path}", Colors.BLUE))
    train_tokens = tokenizer.encode_corpus(train_text)
    train_tokens, train_inserts = insert_dissonance_markers(
        train_tokens,
        tokenizer.non_special_ids,
        tokenizer.dissonance_id,
        DISSONANCE_RATE,
        random.Random(1234),
    )

    print(color_text(f"Test Data: {args.test_path}", Colors.BLUE))
    test_tokens = tokenizer.encode_corpus(test_text)
    test_tokens, test_inserts = insert_dissonance_markers(
        test_tokens,
        tokenizer.non_special_ids,
        tokenizer.dissonance_id,
        DISSONANCE_RATE,
        random.Random(5678),
    )
    print(
        color_text(
            (
                f"Dissonance injections (train/test): "
                f"{train_inserts}/{test_inserts} sequences"
            ),
            Colors.CYAN,
        )
    )

    train_bytes = len(train_text.encode("utf-8"))
    test_bytes = len(test_text.encode("utf-8"))
    dataset = TextDataset(
        train_tokens=train_tokens,
        test_tokens=test_tokens,
        train_text=train_text,
        test_text=test_text,
        train_bytes=train_bytes,
        test_bytes=test_bytes,
        train_path=args.train_path,
        test_path=args.test_path,
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
    )
    model_tag = build_model_tag(config)
    model_dir = pathlib.Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_tag}.pt"
    log_path = model_dir / f"{model_tag}.log"
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

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee((orig_stdout, False), (log_file, True))
    sys.stderr = Tee((orig_stderr, False), (log_file, True))
    start_wall = time.time()
    start_cpu = time.process_time()
    try:
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
            payload = torch.load(model_path, map_location=device)
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
            except RuntimeError as err:
                print(
                    color_text(
                        "Checkpoint load failed (shape mismatch); starting fresh.",
                        Colors.MAGENTA,
                    )
                )
                print(color_text(str(err), Colors.GRAY))

        for cycle in range(1, args.cycles + 1):
            print(color_text(f"\nCycle {cycle}/{args.cycles}", Colors.BLUE))

            train_chars_cycle = (args.block_size + 1) * args.batch_size * args.steps
            test_chars_cycle = (
                (args.block_size + 1)
                * args.batch_size
                * max(1, args.eval_iters * count_eval_calls(args.steps, args.eval_interval))
            )
            dataset.prepare_cycle("train", train_chars_cycle)
            dataset.prepare_cycle("test", test_chars_cycle)

            print(color_text("Training GRCE picoGPT PoC ...", Colors.CYAN))
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
                args.prompt,
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
            print(color_text(f"Saved model to {model_path}", Colors.GREEN))
    finally:
        elapsed_wall = time.time() - start_wall
        elapsed_cpu = time.process_time() - start_cpu
        summary = f"[runtime] wall={elapsed_wall:.2f}s cpu={elapsed_cpu:.2f}s"
        print(summary)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()


if __name__ == "__main__":
    main()
