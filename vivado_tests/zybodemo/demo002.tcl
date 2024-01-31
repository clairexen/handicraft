
read_verilog demo002.v
read_xdc demo002.xdc

synth_design -part xc7z010clg400-2 -top demo002_top
opt_design
place_design
route_design

write_bitstream -force demo002.bit

