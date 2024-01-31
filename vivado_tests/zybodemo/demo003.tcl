
read_verilog demo003.v
read_xdc demo003.xdc

synth_design -part xc7z010clg400-2 -top demo003_top
opt_design
place_design
route_design

write_bitstream -force demo003.bit

