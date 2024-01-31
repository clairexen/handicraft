
create_project -force -part xc7z010clg400-2 system system_prj

set_property ip_repo_paths ip_firfir [current_project]
update_ip_catalog

create_bd_design "system"

create_bd_cell -type ip -vlnv user:user:firfir:1.0 firfir_0

create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 S_AXIS_DATA
set_property CONFIG.TDATA_NUM_BYTES [get_property CONFIG.TDATA_NUM_BYTES [get_bd_intf_pins firfir_0/S_AXIS_DATA]] [get_bd_intf_ports S_AXIS_DATA]
connect_bd_intf_net [get_bd_intf_pins firfir_0/S_AXIS_DATA] [get_bd_intf_ports S_AXIS_DATA]

create_bd_port -dir I -type clk aclk
connect_bd_net [get_bd_pins /firfir_0/aclk] [get_bd_ports aclk]

create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 M_AXIS_DATA
connect_bd_intf_net [get_bd_intf_pins firfir_0/M_AXIS_DATA] [get_bd_intf_ports M_AXIS_DATA]

regenerate_bd_layout
validate_bd_design
save_bd_design

set_property top "system" [current_fileset]
update_compile_order

generate_target all [get_files system.bd]
synth_design
place_design
route_design
