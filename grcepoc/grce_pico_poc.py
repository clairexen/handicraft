"""GRCE proof-of-concept based on the picoGPT Shakespeare demo.

This script keeps the picoGPT spirit of being small and hackable while
adding the Gated Recurrent Context Encoding (GRCE) channel described in
``grce.md``. It trains a very small character-level Transformer on a tiny
Shakespeare sample and shows how the recurrent context vector can be
integrated with a stop-gradient constraint across time.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Data utilities (borrow the spirit of picoGPT's Shakespeare example)
# -----------------------------------------------------------------------------


def load_text_file(path: pathlib.Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Provide --data-path with a plain text file."
        )
    return path.read_text(encoding="utf-8")


class CharTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(list(set(text)))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join(self.itos[int(i)] for i in tokens)


@dataclass
class ShakespeareDataset:
    data: torch.Tensor
    split: float = 0.9

    def __post_init__(self) -> None:
        n = int(self.split * len(self.data))
        self.train = self.data[:n]
        self.val = self.data[n:]

    def get_batch(
        self,
        split: str,
        block_size: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self.train if split == "train" else self.val
        ix = torch.randint(0, len(source) - block_size - 1, (batch_size,))
        x = torch.stack([source[i : i + block_size] for i in ix])
        y = torch.stack([source[i + 1 : i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)


# -----------------------------------------------------------------------------
# Model components (picoGPT-style Transformer blocks + GRCE channel)
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 32
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    context_dim: int = 64
    dropout: float = 0.05


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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        pre_ff = self.ln2(x)
        x = x + self.ff(pre_ff)
        return x, pre_ff


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
        self, idx: torch.Tensor, extra_bias: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        device = idx.device
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = self.drop(tok + pos)
        if extra_bias is not None:
            x = x + extra_bias
        pre_ff = None
        for block in self.blocks:
            x, pre_ff = block(x)
        assert pre_ff is not None
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, pre_ff


class GRCEContextChannel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.reader = nn.Linear(config.context_dim, config.n_embd)
        self.writer = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.context_dim),
            nn.GELU(),
            nn.Linear(4 * config.context_dim, config.context_dim),
        )
        self.gate = nn.Linear(config.n_embd, config.context_dim)

    def project(self, context: torch.Tensor) -> torch.Tensor:
        return self.reader(context)

    def update(self, writer_input: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        candidate = self.writer(writer_input)
        gate = torch.sigmoid(self.gate(writer_input))
        return gate * prev + (1.0 - gate) * candidate


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
        context = torch.zeros(B, self.config.context_dim, device=device)
        logits_steps = []
        for t in range(T):
            prefix = idx[:, : t + 1]
            bias = self.context.project(context.detach())
            extra_bias = torch.zeros(
                B,
                prefix.size(1),
                self.config.n_embd,
                device=device,
                dtype=self.core.tok_emb.weight.dtype,
            )
            extra_bias[:, -1, :] = bias
            logits, pre_ff = self.core(prefix, extra_bias=extra_bias)
            context = self.context.update(pre_ff[:, -1, :], context.detach())
            logits_steps.append(logits[:, -1:, :])
        logits = torch.cat(logits_steps, dim=1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1),
                targets.view(B * T),
            )
        return logits, context, loss


# -----------------------------------------------------------------------------
# Training / generation helpers
# -----------------------------------------------------------------------------


def train_model(
    model: GRCEGPT,
    dataset: ShakespeareDataset,
    device: torch.device,
    steps: int,
    block_size: int,
    batch_size: int,
) -> None:
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for step in range(1, steps + 1):
        xb, yb = dataset.get_batch("train", block_size, batch_size, device)
        logits, _, loss = model.forward_autoreg(xb, yb)
        if loss is None:
            raise RuntimeError("Loss should not be None during training")
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % max(1, steps // 10) == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vb, vy = dataset.get_batch("val", block_size, batch_size, device)
                _, _, vloss = model.forward_autoreg(vb, vy)
            model.train()
            print(
                f"step {step:04d} | train loss {loss.item():.3f} | val loss {vloss.item():.3f}"
            )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=pathlib.Path,
        default=pathlib.Path("data/tiny_shakespeare_sample.txt"),
        help="Path to a plain text corpus (default: tiny sample bundled with repo).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--steps", type=int, default=50, help="Training steps")
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Number of tokens per training sample",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--context-dim",
        type=int,
        default=64,
        help="Dimension of the recurrent context vector",
    )
    parser.add_argument(
        "--generate",
        type=int,
        default=200,
        help="Number of new characters to sample after training",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="ROMEO:",
        help="Prompt used for generation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    random.seed(42)

    text = load_text_file(args.data_path)
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)
    dataset = ShakespeareDataset(data)

    config = ModelConfig(
        vocab_size=len(tokenizer.stoi),
        block_size=args.block_size,
        context_dim=args.context_dim,
    )
    device = torch.device(args.device)
    model = GRCEGPT(config).to(device)

    print("Training GRCE picoGPT PoC ...")
    train_model(model, dataset, device, args.steps, args.block_size, args.batch_size)

    prompt_tokens = tokenizer.encode(args.prompt)
    prompt_tokens = prompt_tokens.unsqueeze(0).to(device)
    generated = generate(model, prompt_tokens, args.generate)
    print("\n---- sample ----")
    print(tokenizer.decode(generated[0].cpu()))


if __name__ == "__main__":
    main()
