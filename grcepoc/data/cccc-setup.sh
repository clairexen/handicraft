#!/bin/bash

if [ ! -f dolma-cccc-filtered-0000.json.gz ]; then
	echo "First, download dolma-cccc-filtered-0000.json.gz from"
	echo "https://huggingface.co/datasets/common-pile/cccc_filtered/tree/main"
	exit 1
fi

set -ev

zcat dolma-cccc-filtered-0000.json.gz | \
jq -r 'select((input_line_number - 1) % 10 != 0) | .text + "\n\n<|----|>\n"' | \
tr A-Z a-z | gzip --rsyncable -9 > cccc-train.txt.gz

zcat dolma-cccc-filtered-0000.json.gz | \
jq -r 'select((input_line_number - 1) % 10 == 0) | .text + "\n\n<|----|>\n"' | \
tr A-Z a-z | gzip --rsyncable -9 > cccc-test.txt.gz

# re-use pre-generated token list for model compatibility
cp simplerwiki_vocab_3000.json cccc_vocab_3000.json

# tokenize corpus
cd ..
.venv/bin/python grce.py --corpus cccc corpus --init

