
read_verilog demo001.v
read_xdc demo001.xdc

synth_design -part xc7z010clg400-2 -top demo001_top
opt_design
place_design
route_design

write_bitstream -force demo001.bit

