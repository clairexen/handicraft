#!/bin/bash

set -ex
. ../../settings.sh

rm -f vivado.log
vivado -mode batch -source dumpwires.tcl -nojournal

