#!/bin/bash

set -vx
vocab=600

if [ ! -f simplestwiki_tokens_test_${vocab}.pt ]; then
	../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
			--input simplestwiki-test.txt.gz --output simplestwiki_tokens_test_${vocab}.pt
fi

if [ ! -f simplestwiki_tokens_train_${vocab}.pt ]; then
	../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
			--input simplestwiki-train.txt.gz --output simplestwiki_tokens_train_${vocab}.pt
fi
