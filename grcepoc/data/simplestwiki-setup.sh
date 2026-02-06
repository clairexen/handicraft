#!/bin/bash

set -evx

if [ ! -f simplestwiki_vocab_500.json ]; then
	../.venv/bin/python ../corpus.py tokenizer --output simplestwiki_vocab_500.json \
			--vocab-size 500 simplestwiki-train.txt.gz
fi

if [ ! -f simplestwiki_tokens_test_500.pt ]; then
	../.venv/bin/python ../corpus.py tokens --tokenizer simplestwiki_vocab_500.json \
			--input simplestwiki-test.txt.gz --output simplestwiki_tokens_test_500.pt
fi

if [ ! -f simplestwiki_tokens_train_500.pt ]; then
	../.venv/bin/python ../corpus.py tokens --tokenizer simplestwiki_vocab_500.json \
			--input simplestwiki-train.txt.gz --output simplestwiki_tokens_train_500.pt
fi
