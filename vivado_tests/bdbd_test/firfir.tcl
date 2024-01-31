
create_project -force -part xc7z010clg400-2  ip_firfir ip_firfir_prj

create_bd_design "design_1"

create_bd_cell -type ip -vlnv xilinx.com:ip:fir_compiler:7.2 fir_compiler_0
copy_bd_objs /  [get_bd_cells {fir_compiler_0}]
connect_bd_intf_net [get_bd_intf_pins fir_compiler_0/M_AXIS_DATA] [get_bd_intf_pins fir_compiler_1/S_AXIS_DATA]

create_bd_port -dir I -type clk aclk
connect_bd_net [get_bd_pins /fir_compiler_0/aclk] [get_bd_ports aclk]

create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 S_AXIS_DATA
set_property CONFIG.TDATA_NUM_BYTES [get_property CONFIG.TDATA_NUM_BYTES [get_bd_intf_pins fir_compiler_0/S_AXIS_DATA]] [get_bd_intf_ports S_AXIS_DATA]
connect_bd_intf_net [get_bd_intf_pins fir_compiler_0/S_AXIS_DATA] [get_bd_intf_ports S_AXIS_DATA]
connect_bd_net -net [get_bd_nets aclk_1] [get_bd_ports aclk] [get_bd_pins fir_compiler_1/aclk]

create_bd_intf_port -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 M_AXIS_DATA
connect_bd_intf_net [get_bd_intf_pins fir_compiler_1/M_AXIS_DATA] [get_bd_intf_ports M_AXIS_DATA]

regenerate_bd_layout
validate_bd_design
save_bd_design


add_files firfir.v
set_property top firfir [current_fileset]
update_compile_order

generate_target all [get_files design_1.bd]

ipx::package_project -root_dir ip_firfir -import_files firfir.v

set_property vendor user [ipx::current_core]
set_property taxonomy /UserIP [ipx::current_core]

ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
close_project
