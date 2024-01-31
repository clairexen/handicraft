create_project -part xc7a35tcpg236-1 -force vivadobug05 vivadobug05

read_xdc vivadobug05.xdc
read_verilog vivadobug05.v

set_property top vivadobug05 [current_fileset]

synth_design
opt_design
place_design
route_design

write_bitstream -force vivadobug05.bit

# open_hw
# connect_hw_server
# open_hw_target [lindex [get_hw_targets] 0]
# set_property PROGRAM.FILE vivadobug05.bit [lindex [get_hw_devices] 0]
# program_hw_devices [lindex [get_hw_devices] 0]
