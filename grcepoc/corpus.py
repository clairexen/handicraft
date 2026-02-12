#!/usr/bin/env python3
"""Utility for tokenizer creation and corpus pre-tokenization."""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import time
from typing import Iterable

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = """
<|----|> <|//|> <|tokipona:> <|english:> </english:> </tokipona:>
<|lq:> <:lq|> <|hq:> <:hq|> <|!:> <:!|> <|?:> <:?|> <|*:> <:*|>
<|-:> <:-|> <|=:> <:=|> <|/:> <:/|> <|@:> <:@|> <|reject|> <|think|>
<|p7|> <|p6|> <|p5|> <|p4|> <|p3|> <|p2|> <|p1|> <|p0|>
<|reserved0|> <|reserved1|> <|reserved2|> <|reserved3|>
<|reserved4|> <|reserved5|> <|reserved6|> <|reserved7|>
""".split()

DEFAULT_LIMIT = 1 * 1024 * 1024  # 1 MiB
ENCODE_BATCH_SIZE = 64  # number of text chunks per encode_batch call


def _open_text(path: pathlib.Path):
    """Yield a text-mode file handle for .txt or .txt.gz inputs."""

    if not path.exists():
        raise FileNotFoundError(f"Text file {path} not found")
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    if suffix == ".txt":
        return path.open("rt", encoding="utf-8", errors="ignore")
    raise ValueError(f"Expected .txt or .txt.gz input, got {path}")


def read_limited_text(path: pathlib.Path, limit: int | None) -> str:
    """Load up to ``limit`` bytes from ``path`` (.txt or .txt.gz)."""

    with _open_text(path) as handle:
        text = handle.read(limit)
    return text


def iter_training_text(paths: list[pathlib.Path], limit: int) -> Iterable[str]:
    """Yield limited text snippets for tokenizer training."""

    for path in paths:
        data = read_limited_text(path, limit)
        if data:
            yield data


def write_json(path: pathlib.Path, data: dict) -> None:
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def build_tokenizer(args: argparse.Namespace) -> int:
    paths = [pathlib.Path(spec) for spec in args.inputs]
    limit = args.limit_bytes if args.limit_bytes is not None else DEFAULT_LIMIT
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    byte_alphabet = ByteLevel.alphabet()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        initial_alphabet=byte_alphabet,
        special_tokens=list(SPECIAL_TOKENS),
    )
    tokenizer.train_from_iterator(iter_training_text(paths, limit), trainer=trainer)
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    tokenizer_json = json.loads(tokenizer.to_str())
    write_json(args.output, tokenizer_json)
    print(f"Wrote tokenizer JSON to {args.output}")
    return 0


def encode_corpus(args: argparse.Namespace) -> int:
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    chunk_chars = args.chunk_chars
    token_ids: list[int] = []
    batch: list[str] = []
    total_bytes = 0
    saw_text = False
    total_tokens = 0
    next_log_tokens = 1 << 16
    start_time = time.monotonic()

    def maybe_log_progress() -> None:
        nonlocal next_log_tokens
        if total_tokens < next_log_tokens:
            return
        while total_tokens >= next_log_tokens:
            elapsed = time.monotonic() - start_time
            minutes = elapsed / 60 if elapsed > 0 else 0.0
            bytes_per_min = total_bytes / minutes if minutes else 0.0
            tokens_per_min = total_tokens / minutes if minutes else 0.0
            print(
                f"[encode tokens] elapsed={elapsed:.1f}s bytes={total_bytes} tokens={total_tokens} "
                f"MB/min={bytes_per_min/1e6:.2f} MT/min={tokens_per_min/1e6:.2f}"
            )
            next_log_tokens <<= 1

    def flush_batch() -> None:
        nonlocal total_tokens
        if not batch:
            return
        encodings = tokenizer.encode_batch(batch)
        for encoding in encodings:
            ids = encoding.ids
            token_ids.extend(ids)
            total_tokens += len(ids)
            maybe_log_progress()
        batch.clear()

    with _open_text(args.input) as handle:
        while True:
            piece = handle.read(chunk_chars)
            if not piece:
                break
            saw_text = True
            batch.append(piece)
            total_bytes += len(piece.encode("utf-8"))
            if len(batch) >= ENCODE_BATCH_SIZE:
                flush_batch()
        flush_batch()

    if not saw_text:
        raise ValueError(f"Input corpus {args.input} is empty")

    tensor = torch.tensor(token_ids, dtype=torch.uint16)
    payload = {"tokens": tensor, "bytes": total_bytes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"Wrote {tensor.numel()} tokens to {args.output} using tokenizer {args.tokenizer}"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tok = subparsers.add_parser(
        "tokenizer",
        help="Build a tokenizer JSON from limited *.txt/.txt.gz inputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    tok.add_argument("--output", type=pathlib.Path, required=True)
    tok.add_argument("--vocab-size", type=int, required=True)
    tok.add_argument(
        "--limit-bytes",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum bytes to consume from each input file",
    )
    tok.add_argument("inputs", nargs="+", help="*.txt or *.txt.gz sources", type=str)
    tok.set_defaults(func=build_tokenizer)

    enc = subparsers.add_parser(
        "tokens",
        help="Encode a .txt or .txt.gz file into a .pt token cache",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    enc.add_argument("--tokenizer", type=pathlib.Path, required=True)
    enc.add_argument("--input", type=pathlib.Path, required=True)
    enc.add_argument("--output", type=pathlib.Path, required=True)
    enc.add_argument(
        "--chunk-chars",
        type=int,
        default=2048,
        help="Number of characters to encode per chunk",
    )
    enc.set_defaults(func=encode_corpus)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
