#!/bin/bash
set -ex
yosys equiv.ys
yosys -l wormux1.log -p 'synth_ice40 -top wormux1' wormux.v
yosys -l wormux2.log -p 'synth_ice40 -top wormux2' wormux.v
yosys -l wormux3.log -p 'synth_ice40 -top wormux3' wormux.v
