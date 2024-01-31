#!/bin/bash

set -ex
. ../settings.sh

rm -f vivado.log
vivado -mode batch -source playground.tcl -nojournal

for f in test[0-9][0-9][0-9].bit; do
	../tools/bitread -xzo ${f%.bit}.asc < $f
done

# ../tools/bitmatch -o ../database/lutinit.cbits -v test[0-9][0-9][0-9].tags | tee lutbits.log
../tools/bitcorr -m -o corrbits.cbits -v test[0-9][0-9][0-9].tags | tee corrbits.log

