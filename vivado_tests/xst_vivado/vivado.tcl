
read_verilog top.v
read_edif wxzip.ndf

synth_design -part xc7k70t -top top

opt_design
place_design
route_design

