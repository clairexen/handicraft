#!/bin/bash

set -ex

git clone https://github.com/pulp-platform/pulpino
cd pulpino
./update-ips.py

cd ips/riscv
yosys ../../../ri5cy.ys
cp ri5cy.v ../../..

if false; then
	cd verilator-model
	make
	./testbench
	cd ..
fi

cd ../../..
iverilog -o testbench testbench.v ri5cy.v
./testbench
