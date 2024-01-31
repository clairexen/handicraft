#!/bin/bash

set -ex

cd myip1
vivado -mode batch -source build.tcl

cd ../myip2
vivado -mode batch -source build.tcl

cd ../mytop1
vivado -mode batch -source build.tcl

cd ../mytop2
vivado -mode batch -source build.tcl
