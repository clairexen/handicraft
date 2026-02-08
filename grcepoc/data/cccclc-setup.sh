#!/bin/bash

set -ex
vocab=6000
for volume; do
	if [ ! -f dolma-cccc-filtered-${volume}.json.gz ]; then
		# https://huggingface.co/datasets/common-pile/cccc_filtered/tree/main
		wget "https://huggingface.co/datasets/common-pile/cccc_filtered/resolve/main/dolma-cccc-filtered-${volume}.json.gz"
	fi

	if [ ! -f cccclc-${volume}-train.txt.gz ]; then
		../.venv/bin/python cccclc-split.py ${volume}
	fi

	if [ ! -f cccclc-${volume}_tokens_test_${vocab}.pt ]; then
		../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
				--input cccclc-${volume}-test.txt.gz --output cccclc-${volume}_tokens_test_${vocab}.pt
	fi

	if [ ! -f cccclc-${volume}_tokens_train_${vocab}.pt ]; then
		../.venv/bin/python ../corpus.py tokens --tokenizer vocab_${vocab}.json \
				--input cccclc-${volume}-train.txt.gz --output cccclc-${volume}_tokens_train_${vocab}.pt
	fi
done
