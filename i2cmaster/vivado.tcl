read_verilog i2c_master.v i2c_master_bit_ctrl.v i2c_master_byte_ctrl.v
read_xdc vivado.xdc

synth_design -part xc7k70t-fbg676 -top i2c_master -debug_log -verbose
# opt_design
# place_design
# phys_opt_design

report_utilization -file vivado.rpt -hierarchical
report_timing -append -file vivado.rpt
