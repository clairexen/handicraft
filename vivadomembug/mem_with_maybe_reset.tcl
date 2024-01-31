read_verilog mem_with_maybe_reset.v
synth_design -part xc7k70t-fbg676 -top mem_with_maybe_reset
opt_design
report_utilization
