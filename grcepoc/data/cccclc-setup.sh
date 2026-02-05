#!/bin/bash

set -evx
volume=${1:-0000}

if [ ! -f dolma-cccc-filtered-${volume}.json.gz ]; then
	# https://huggingface.co/datasets/common-pile/cccc_filtered/tree/main
	wget "https://huggingface.co/datasets/common-pile/cccc_filtered/resolve/main/dolma-cccc-filtered-${volume}.json.gz"
	exit 1
fi

if [ ! -f cccclc-${volume}-train.txt.gz ]; then
	python cccclc-split.py ${volume}
fi

if [ ! -f cccclc-${volume}_tokens_train_5000.pt ]; then
	cp cccclc_vocab_5000.json cccclc-${volume}_vocab_5000.json
	( cd ..; .venv/bin/python grce.py --corpus cccclc-${volume} corpus --init; )
fi
