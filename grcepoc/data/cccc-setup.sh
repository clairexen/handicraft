#!/bin/bash

if [ ! -f dolma-cccc-filtered-0000.json.gz ]; then
	echo "First, download dolma-cccc-filtered-0000.json.gz from"
	echo "https://huggingface.co/datasets/common-pile/cccc_filtered/tree/main"
	exit 1
fi

set -ev

# this will take a little while...
python cccc-split.py

# re-use pre-generated token list for model compatibility
cp simplerwiki_vocab_3000.json cccc_vocab_3000.json

# tokenize corpus
cd ..
.venv/bin/python grce.py --corpus cccc corpus --init
