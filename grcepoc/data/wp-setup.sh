#!/bin/bash

set -ex
vocab=32000

lang=simple
if [ ! -f wp-${lang}-train.txt.gz ]; then
	../.venv/bin/python wp-split.py simple
fi
if [ ! -f wp-${lang}_tokens_test_${vocab}.pt ]; then
	../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
			--input wp-${lang}-test.txt.gz --output wp-${lang}_tokens_test_${vocab}.pt
fi
if [ ! -f wp-${lang}_tokens_train_${vocab}.pt ]; then
	../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
			--input wp-${lang}-train.txt.gz --output wp-${lang}_tokens_train_${vocab}.pt
fi

lang=en
for volume in 0 1 2 3 4 5 6 7; do
	if [ ! -f wp-${lang}-000${volume}-train.txt.gz ]; then
		../.venv/bin/python wp-split.py en 8 ${volume}
	fi

	if [ ! -f wp-${lang}-000${volume}_tokens_test_${vocab}.pt ]; then
		../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
				--input wp-${lang}-000${volume}-test.txt.gz --output wp-${lang}-000${volume}_tokens_test_${vocab}.pt
	fi

	if [ ! -f wp-${lang}-000${volume}_tokens_train_${vocab}.pt ]; then
		../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
				--input wp-${lang}-000${volume}-train.txt.gz --output wp-${lang}-000${volume}_tokens_train_${vocab}.pt
	fi
done
