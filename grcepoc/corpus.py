#!/usr/bin/env python3
"""Utility for tokenizer creation and corpus pre-tokenization."""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
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
""".split()



DEFAULT_LIMIT = 1 * 1024 * 1024  # 1 MiB


def read_limited_text(path: pathlib.Path, limit: int | None) -> str:
    """Load up to ``limit`` bytes from ``path`` (expects *.txt.gz)."""

    if not path.exists():
        raise FileNotFoundError(f"Text file {path} not found")
    if path.suffix != ".gz":
        raise ValueError(f"Expected .txt.gz input, got {path}")
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
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


def read_full_text(path: pathlib.Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Text file {path} not found")
    if path.suffix != ".gz":
        raise ValueError(f"Expected .txt.gz input, got {path}")
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        return handle.read()


def encode_corpus(args: argparse.Namespace) -> int:
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    text = read_full_text(args.input)
    if not text:
        raise ValueError(f"Input corpus {args.input} is empty")
    chunk = args.chunk_chars
    token_ids: list[int] = []
    for offset in range(0, len(text), chunk):
        piece = text[offset : offset + chunk]
        token_ids.extend(tokenizer.encode(piece).ids)
    tensor = torch.tensor(token_ids, dtype=torch.uint16)
    payload = {"tokens": tensor, "bytes": len(text.encode("utf-8"))}
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
        help="Build a tokenizer JSON from limited *.txt.gz inputs",
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
    tok.add_argument("inputs", nargs="+", help="*.txt.gz sources", type=str)
    tok.set_defaults(func=build_tokenizer)

    enc = subparsers.add_parser(
        "tokens",
        help="Encode a .txt.gz file into a .pt token cache",
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
