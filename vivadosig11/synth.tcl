# run with: vivado -mode batch -source synth.tcl

read_verilog picorv32.v
read_verilog top.v

synth_design -part xc7k70t-fbg676 -top top
opt_design -sweep -propconst -resynth_seq_area
opt_design -directive ExploreSequentialArea

report_utilization
report_timing
