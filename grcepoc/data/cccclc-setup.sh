#!/bin/bash

set -evx
volume=${1:-0000}

if [ ! -f dolma-cccc-filtered-${volume}.json.gz ]; then
	# https://huggingface.co/datasets/common-pile/cccc_filtered/tree/main
	wget "https://huggingface.co/datasets/common-pile/cccc_filtered/resolve/main/dolma-cccc-filtered-${volume}.json.gz"
fi

if [ ! -f cccclc-${volume}-train.txt.gz ]; then
	../.venv/bin/python cccclc-split.py ${volume}
fi

if [ ! -f cccclc-${volume}_tokens_train_5000.pt ]; then
	../.venv/bin/python ../corpus.py tokens --tokenizer cccclc_vocab_5000.json \
			--input cccclc-${volume}-test.txt.gz --output cccclc-${volume}_tokens_test_5000.pt
fi
