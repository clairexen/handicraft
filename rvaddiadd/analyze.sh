#!/bin/bash
mkdir -p results
for f in $(find database/ -type f -size +100k -size -100M -name "*.txt" | sort -R); do
	n="${f%.txt}"
	n="${n#database/}"
	echo "all:: results/$n.txt"
	echo "results/$n.txt: database/$n.txt"
	echo "	python3 analyze.py database/$n.txt > results/$n.tmp"
	echo "	mv results/$n.tmp results/$n.txt"
done > analyze.mk
