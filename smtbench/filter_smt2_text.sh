#!/bin/bash
# Quick & Dirty text filter for SMT2 benchmarks
# (remove comments and statements such as get-value)
sed 's, *;.*,,' | tr -s ' ' | grep . | grep -v get-value
