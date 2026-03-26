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

DEFAULT_LIMIT: int | None = None  # Unlimited by default
ENCODE_BATCH_SIZE = 64  # number of text chunks per encode_batch call

TOKIPONA_WORDS = """
a akesi ala alasa ale anpa ante anu apeja awen e en epiku esun ijo
ike ilo insa jaki jan jasima jelo jo kala kalama kama kasi ken kepeken
kijetesantakalu kili kipisi kiwen ko kokosila kon kule kulupu kute la
lanpan lape laso lawa leko len lete li lili linja lipu loje lon luka
lukin lupa ma majuna mama mani meli meso mi mije misikeke moku moli
monsi monsuta mu mun musi mute nanpa nasa nasin nena ni nimi noka o oke
olin ona open pakala pali palisa pan pana pi pilin pimeja pini pipi poka
poki pona powe pu sama seli selo seme sewi sijelo sike sin sina sinpin
sitelen soko sona soweli su suli suno supa suwi tan taso tawa telo tenpo
toki tomo tonsi tu unpa uta utala walo wan waso wawa weka wile
""".split()

def ensure_tokipona_vocab(tokenizer_json: dict) -> None:
    vocab = tokenizer_json.get("model", {}).get("vocab", {})
    added_tokens = tokenizer_json.setdefault("added_tokens", [])
    existing_added = {entry.get("content") for entry in added_tokens}
    used_ids = set(vocab.values()) | {entry.get("id") for entry in added_tokens if isinstance(entry.get("id"), int)}
    next_id = (max(used_ids) + 1) if used_ids else 0

    def add_token(token: str) -> None:
        nonlocal next_id
        if token in vocab or token in existing_added:
            return
        added_tokens.append(
            {
                "id": next_id,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "special": False,
                "normalized": False,
            }
        )
        existing_added.add(token)
        next_id += 1

    for word in TOKIPONA_WORDS:
        add_token(word)
        add_token(f"Ġ{word}")

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
        size = -1 if limit is None else limit
        text = handle.read(size)
    return text


def iter_training_text(
    entries: list[tuple[pathlib.Path, int | None, int]],
    default_limit: int | None,
) -> Iterable[str]:
    """Yield limited text snippets for tokenizer training."""

    def sanitize(text: str) -> str:
        sanitized = text
        # Replace reserved tokens with whitespace so the trainer cannot learn partial merges.
        for token in SPECIAL_TOKENS:
            if token in sanitized:
                sanitized = sanitized.replace(token, " ")
        return sanitized

    for path, local_limit, weight in entries:
        limit = local_limit if local_limit is not None else default_limit
        data = read_limited_text(path, limit)
        if data:
            repeat = max(1, weight)
            sanitized = sanitize(data)
            for _ in range(repeat):
                yield sanitized


def write_json(path: pathlib.Path, data: dict) -> None:
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def build_tokenizer(args: argparse.Namespace) -> int:
    entries = [parse_limited_input(spec) for spec in args.inputs]
    limit = args.limit_bytes if args.limit_bytes is not None else DEFAULT_LIMIT
    tokipona_reserve = max(0, args.tokipona or 0)
    if tokipona_reserve >= args.vocab_size:
        raise ValueError("--tokipona reserve must be smaller than --vocab-size")
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    byte_alphabet = ByteLevel.alphabet()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size - tokipona_reserve,
        min_frequency=2,
        initial_alphabet=byte_alphabet,
        special_tokens=list(SPECIAL_TOKENS),
    )
    tokenizer.train_from_iterator(iter_training_text(entries, limit), trainer=trainer)
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    tokenizer_json = json.loads(tokenizer.to_str())
    if tokipona_reserve > 0:
        ensure_tokipona_vocab(tokenizer_json)
    vocab = tokenizer_json.get("model", {}).get("vocab", {})
    added_tokens = tokenizer_json.get("added_tokens", [])
    vocab_count = len(vocab)
    extra_added = sum(1 for entry in added_tokens if entry.get("content") not in vocab)
    total_tokens = vocab_count + extra_added
    if total_tokens != args.vocab_size:
        print(
            f"Warning: tokenizer produced {total_tokens} tokens (model={vocab_count}, unique added={extra_added})"
            f" but --vocab-size requested {args.vocab_size}",
        )
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
                f"[encode tokens] elapsed={elapsed:.1f}s MB={total_bytes/1e6:.2f} MT={total_tokens/1e6:.2f} "
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

    dtype = torch.uint32 if args.vocab_size > (1 << 16) else torch.uint16
    tensor = torch.tensor(token_ids, dtype=dtype)
    payload = {"tokens": tensor, "bytes": total_bytes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"Wrote {tensor.numel()} tokens to {args.output} using tokenizer {args.tokenizer}"
    )
    return 0


def parse_limit_bytes(text: str) -> int:
    text = text.strip()
    if not text:
        raise ValueError("Limit cannot be empty")
    suffixes = {"k": 1024, "K": 1024, "m": 1024 * 1024, "M": 1024 * 1024}
    suffix = text[-1]
    if suffix in suffixes:
        base = float(text[:-1])
        return int(base * suffixes[suffix])
    return int(text)


def parse_limited_input(spec: str) -> tuple[pathlib.Path, int | None, int]:
    path_text = spec
    limit: int | None = None
    weight = 1
    while True:
        idx = path_text.rfind(":")
        if idx == -1:
            break
        candidate = path_text[idx + 1 :]
        if not candidate:
            break
        if (candidate.endswith("x") or candidate.endswith("X")) and candidate[:-1].isdigit():
            value = int(candidate[:-1])
            if value > 0:
                weight = value
            path_text = path_text[:idx]
            continue
        try:
            limit = parse_limit_bytes(candidate)
            path_text = path_text[:idx]
            continue
        except ValueError:
            break
    return pathlib.Path(path_text), limit, weight


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
        "--tokipona",
        type=int,
        default=0,
        metavar="RESERVE",
        help=(
            "Reserve RESERVE slots (must be < vocab size) to force every Toki Pona word "
            "with and without a leading space into the vocabulary."
        ),
    )
    tok.add_argument(
        "--limit-bytes",
        type=parse_limit_bytes,
        default=DEFAULT_LIMIT,
        help=(
            "Maximum bytes to consume from each input file (supports k/M suffix). "
            "Defaults to reading the entire file."
        ),
    )
    tok.add_argument(
        "inputs",
        nargs="+",
        help=(
            "*.txt or *.txt.gz sources. Append :<limit> and/or :<Nx> per file to override"
            " defaults (e.g. book.txt:512k:3x or notes.txt:2x)."
        ),
        type=str,
    )
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
