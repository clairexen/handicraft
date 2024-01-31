#!/bin/bash

set -ex
. ../settings.sh

rm -f vivado.log
vivado -mode batch -source playground.tcl -nojournal

for f in test[0-9][0-9][0-9].bit; do
	../tools/bitread -xzo ${f%.bit}.asc < $f
done

../tools/bitcorr -m -o corrbits_m.cbits -v test[0-9][0-9][0-9].tags | grep -v ^Reading | tee corrbits_m.log
../tools/bitcorr -o corrbits.cbits -v test[0-9][0-9][0-9].tags | grep -v ^Reading | tee corrbits.log
