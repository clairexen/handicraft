# vivado -mode batch -source demo1.tcl
create_project -force -part xc7a35tcpg236-1 demo1
read_verilog demo1.v
synth_design -top demo1
opt_design
place_design
route_design
report_utilization
