#!/bin/bash
set -ex
gcc -o aa64lst -Wall -Wextra -O2 aa64lst.c
./aa64lst | sort > aa64lst.txt
python3 rvb.py
