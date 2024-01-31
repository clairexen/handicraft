#!/bin/bash
set -ex
. ../../settings.sh
mkdir ../../database
vivado -mode batch -source makejson.tcl -nolog -nojournal
python3 makehtml.py > ../../database/tilegrid.html
