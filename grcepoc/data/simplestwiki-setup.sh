#!/bin/bash

set -vx
for vocab in 600 30000; do

if [ ! -f simplestwiki_tokens_test_${vocab}.pt ]; then
	../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
			--input simplestwiki-test.txt.gz --output simplestwiki_tokens_test_${vocab}.pt
fi

if [ ! -f simplestwiki_tokens_train_${vocab}.pt ]; then
	../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
			--input simplestwiki-train.txt.gz --output simplestwiki_tokens_train_${vocab}.pt
fi

done
