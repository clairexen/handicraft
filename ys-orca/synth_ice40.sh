#!/bin/bash
set -ex
yosys_bin=${HOME}/Work/yosys-verific/yosys
[ -d orca ] || git clone https://github.com/VectorBlox/orca orca
cd orca/rtl
sed -i '/AVALON_ENABLE.*:=/ s/0;/1;/;' orca.vhd
$yosys_bin -v2 -l synth_ice40.log ../../synth_ice40.ys
mv synth_ice40.{v,il,blif,log} ../..
