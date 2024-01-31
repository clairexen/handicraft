
create_project -part xc7z010clg400-2 -in_memory


### Setup IP Repository

file delete -force demo004_iprepo
file mkdir demo004_iprepo

set_property IP_REPO_PATHS demo004_iprepo [current_fileset]
update_ip_catalog -rebuild

update_ip_catalog -add_ip demo004_generator_101.zip -repo_path demo004_iprepo
update_ip_catalog -add_ip demo004_consumer_101.zip -repo_path demo004_iprepo


### Create Block Design

create_bd_design system

create_bd_cell -type ip -vlnv clifford.at:demo004:generator my_generator
set_property CONFIG.limit 200000000 [get_bd_cells my_generator]

create_bd_cell -type ip -vlnv clifford.at:demo004:consumer my_consumer

create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:5.1 clk_wiz
set_property -dict {
	CONFIG.CLKOUT1_REQUESTED_OUT_FREQ	200.000
	CONFIG.USE_RESET			false
} [get_bd_cells clk_wiz]

create_bd_port -dir I -type clk clk
set_property CONFIG.FREQ_HZ 125000000 [get_bd_ports clk]

create_bd_port -dir O led

connect_bd_net [get_bd_pins clk_wiz/clk_out1] [get_bd_pins my_generator/aclk] [get_bd_pins my_consumer/aclk]
connect_bd_net [get_bd_pins clk_wiz/locked] [get_bd_pins my_generator/aresetn] [get_bd_pins my_consumer/aresetn]

connect_bd_net [get_bd_ports clk] [get_bd_pins clk_wiz/clk_in1]
connect_bd_net [get_bd_ports led] [get_bd_pins my_consumer/state]
connect_bd_intf_net [get_bd_intf_pins my_generator/m_axis] [get_bd_intf_pins my_consumer/s_axis]

regenerate_bd_layout
validate_bd_design


### Synthesis

read_xdc demo004.xdc
generate_target all [get_files system.bd]

synth_design -top system
opt_design
place_design
route_design

write_bitstream -force demo004.bit

