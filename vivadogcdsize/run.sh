#!/bin/bash
yosys -l gcd_hier.yslog -p 'rename gcd gcd_hier; synth_xilinx -top gcd_hier -edif gcd_hier.edn' gcd.sv
yosys -l gcd_flat.yslog -p 'rename gcd gcd_flat; synth_xilinx -top gcd_flat -edif gcd_flat.edn -flatten' gcd.sv
vivado -mode tcl -source gcd_hier.tcl -nojournal -log gcd_hier.log
vivado -mode tcl -source gcd_flat.tcl -nojournal -log gcd_flat.log
