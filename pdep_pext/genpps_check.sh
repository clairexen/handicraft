#!/bin/bash

set -ex

{
	python3 genpps.py -m gold -n 16 -i 2 -o 6 -IO -s
	python3 genpps.py -m gate1 -n 16 -i 2 -o 6 -IO
	python3 genpps.py -m gate2 -n 16 -i 2 -o 6 -IO -a
} > genpps_check.v

yosys << EOT
read_verilog genpps_check.v
prep
miter -equiv -flatten gold gate1 miter1
miter -equiv -flatten gold gate2 miter2
sat -verify -prove trigger 0 miter1
sat -verify -prove trigger 0 miter2
EOT

