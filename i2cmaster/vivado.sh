#!/bin/bash
(
export FORCE_PS1="[Vivado] $PS1"
. /opt/Xilinx/Vivado/2021.2/settings64.sh
"$@"
)
