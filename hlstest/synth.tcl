read_verilog mycore.v
read_xdc synth.xdc

synth_design -part xc7z030-ffg676-1 -top mycore
opt_design
place_design
phys_opt_design
route_design

report_utilization
report_timing
