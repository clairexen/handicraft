#!/bin/bash

set -ex
vocab=32000

lang=simple
if [ ! -f wikipedia-${lang}-train.txt.gz ]; then
	../.venv/bin/python wikipedia-split.py simple
fi
if [ ! -f wikipedia-${lang}_tokens_test_${vocab}.pt ]; then
	../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
			--input wikipedia-${lang}-test.txt.gz --output wikipedia-${lang}_tokens_test_${vocab}.pt
fi
if [ ! -f wikipedia-${lang}_tokens_train_${vocab}.pt ]; then
	../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
			--input wikipedia-${lang}-train.txt.gz --output wikipedia-${lang}_tokens_train_${vocab}.pt
fi

lang=en
for volume in 0 1 2 3 4 5 6 7; do
	if [ ! -f wikipedia-${lang}-000${volume}-train.txt.gz ]; then
		../.venv/bin/python wikipedia-split.py en 8 ${volume}
	fi

	if [ ! -f wikipedia-${lang}-000${volume}_tokens_test_${vocab}.pt ]; then
		../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
				--input wikipedia-${lang}-000${volume}-test.txt.gz --output wikipedia-${lang}-000${volume}_tokens_test_${vocab}.pt
	fi

	if [ ! -f wikipedia-${lang}-000${volume}_tokens_train_${vocab}.pt ]; then
		../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
				--input wikipedia-${lang}-000${volume}-train.txt.gz --output wikipedia-${lang}-000${volume}_tokens_train_${vocab}.pt
	fi
done
