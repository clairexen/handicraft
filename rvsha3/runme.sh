#!/bin/bash
set -ex

gcc -Wall -o keccak_ref keccak_ref.c
./keccak_ref > keccak_ref.out
cmp keccak_ref.out expected.txt

gcc -Wall -o keccak_rv64b keccak_rv64b.c
./keccak_rv64b > keccak_rv64b.out
diff -u keccak_rv64b.out expected.txt

: OK
