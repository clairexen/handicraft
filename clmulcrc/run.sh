#!/bin/bash
set -ex
python3 crcfold.py
gcc -o clmulcrc -Wall -Os clmulcrc.c -lz
./clmulcrc
: OKAY
