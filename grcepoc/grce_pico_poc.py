"""GRCE proof-of-concept based on the picoGPT Shakespeare demo.

This script keeps the picoGPT spirit of being small and hackable while
adding the Gated Recurrent Context Encoding (GRCE) channel described in
``grce.md``. It trains a tiny character-level Transformer on the bundled
Simple English Wikipedia split and shows how the recurrent context vector can
be integrated with a stop-gradient constraint across time.
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
        raise FileNotFoundError(f"Could not find {path}. Provide a text file path.")
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
class TextDataset:
    train: torch.Tensor
    test: torch.Tensor

    def get_batch(
        self,
        split: str,
        block_size: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split {split!r}")
        source = self.train if split == "train" else self.test
        if len(source) <= block_size:
            raise ValueError(
                f"Split {split} is too small for block size {block_size}."
            )
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
    sample_prompt: torch.Tensor,
    sample_chars: int,
    tokenizer: CharTokenizer,
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
            print(
                f"step {step:04d} | train loss {split_losses['train']:.3f} | "
                f"test loss {split_losses['test']:.3f} | sample: {sample_text}"
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
        "--eval-interval",
        type=int,
        default=25,
        help="How often to run train/test evaluation steps.",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=5,
        help="How many mini-batches to average for evaluation losses.",
    )
    parser.add_argument(
        "--max-train-chars",
        type=int,
        default=0,
        help="Optional limit on how many characters of the training file to use.",
    )
    parser.add_argument(
        "--max-test-chars",
        type=int,
        default=0,
        help="Optional limit on how many characters of the test file to use.",
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
        default="Bigotry is",
        help="Prompt used for generation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    random.seed(42)

    train_text = load_text_file(args.train_path)
    test_text = load_text_file(args.test_path)
    if args.max_train_chars > 0:
        train_text = train_text[: args.max_train_chars]
    if args.max_test_chars > 0:
        test_text = test_text[: args.max_test_chars]
    tokenizer = CharTokenizer(train_text + test_text)
    train_tokens = tokenizer.encode(train_text)
    test_tokens = tokenizer.encode(test_text)
    dataset = TextDataset(train_tokens, test_tokens)

    device = torch.device(args.device)
    try:
        prompt_tokens = tokenizer.encode(args.prompt)
    except KeyError as exc:  # pragma: no cover - user misconfiguration
        raise ValueError(
            "Prompt contains characters outside the tokenizer vocabulary. "
            "Choose a simpler prompt or extend the dataset."
        ) from exc
    prompt_tokens = prompt_tokens.unsqueeze(0).to(device)

    config = ModelConfig(
        vocab_size=len(tokenizer.stoi),
        block_size=args.block_size,
        context_dim=args.context_dim,
    )
    model = GRCEGPT(config).to(device)

    print("Training GRCE picoGPT PoC ...")
    train_model(
        model,
        dataset,
        device,
        args.steps,
        args.block_size,
        args.batch_size,
        args.eval_interval,
        args.eval_iters,
        prompt_tokens,
        args.generate,
        tokenizer,
    )


if __name__ == "__main__":
    main()
