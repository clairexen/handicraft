#!/bin/bash
set -ex

for ((i=0; i<${1:-1}; i++)); do
	pdflatex xposit
	bibtex xposit
done
