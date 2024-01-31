# vivado -mode batch -source demo2.tcl
create_project -force -part xc7a35tcpg236-1 demo2
read_verilog demo2.v
synth_design -top demo2
opt_design
place_design
route_design
report_utilization
