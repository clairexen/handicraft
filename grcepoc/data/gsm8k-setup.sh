#!/bin/bash

set -ex
vocab=30000

if [ ! -f gsm8k-test.jsonl.gz ]; then
	wget -O gsm8k-test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/refs/heads/master/grade_school_math/data/test.jsonl
	gzip gsm8k-test.jsonl
fi
if [ ! -f gsm8k-train.jsonl.gz ]; then
	wget -O gsm8k-train.jsonl https://raw.githubusercontent.com/openai/grade-school-math/refs/heads/master/grade_school_math/data/train.jsonl
	gzip gsm8k-train.jsonl
fi

if [ ! -f gsm8k-test.txt.gz ]; then
	python gsm8k-convert.py --output gsm8k-test.txt.gz gsm8k-test.jsonl.gz
fi

if [ ! -f gsm8k-train.txt.gz ]; then
	python gsm8k-convert.py --output gsm8k-train.txt.gz gsm8k-train.jsonl.gz
fi

if [ ! -f gsm8k_tokens_test_${vocab}.pt ]; then
	../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
			--input gsm8k-test.txt.gz --output gsm8k_tokens_test_${vocab}.pt
fi
if [ ! -f gsm8k_tokens_train_${vocab}.pt ]; then
	../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
			--input gsm8k-train.txt.gz --output gsm8k_tokens_train_${vocab}.pt
fi
