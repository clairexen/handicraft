
create_project -part xc7z010clg400-2 -in_memory


### Setup IP Repository

file delete -force demo006_iprepo
file mkdir demo006_iprepo

set_property IP_REPO_PATHS demo006_iprepo [current_fileset]
update_ip_catalog -rebuild

update_ip_catalog -add_ip demo006_minigpio_101.zip -repo_path demo006_iprepo


### Create Block Design

create_bd_design system

create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 ps7
set_property -dict [list CONFIG.PCW_IMPORT_BOARD_PRESET {demo006_zynq_def.xml}] [get_bd_cells ps7]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
	-config {make_external "FIXED_IO, DDR" Master "Disable" Slave "Disable"}  [get_bd_cells ps7]

create_bd_cell -type ip -vlnv clifford.at:demo006:minigpio:1.01 gpio

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/ps7/M_AXI_GP0" Clk "Auto"}  [get_bd_intf_pins gpio/s_axi]
set_property offset 0x50000000 [get_bd_addr_segs {ps7/Data/SEG_gpio_reg0}]

create_bd_port -dir I -from 31 -to 0 -type data sw
create_bd_port -dir O -from 31 -to 0 -type data led

connect_bd_net [get_bd_ports sw] [get_bd_pins gpio/gpio_i]
connect_bd_net [get_bd_ports led] [get_bd_pins gpio/gpio_o]

regenerate_bd_layout
validate_bd_design


### Synthesis

read_xdc demo006.xdc
read_verilog demo006_wrapper.v
set_property top system_wrapper [current_fileset]

generate_target all [get_files system.bd]
export_hardware -dir demo006.sdk [get_files system.bd]

synth_design
opt_design
place_design
route_design

write_bitstream -force demo006.bit

