# vivado -mode batch -nolog -nojournal -source vivado_make_ip.tcl

create_project -ip -in_memory
set ip [ipx::create_core example.com example_cores myromtable 1.0]

set_property description {Example IP for $readmemh() usage in IP HDL} $ip
set_property display_name ip_myromtable $ip
set_property root_directory ip_myromtable $ip
set_property taxonomy {{/Example}} $ip

set_property supported_families {
	zynq	Production
} $ip

set grp [ipx::add_file_group -type verilog:simulation {} $ip]
set_property model_name myromtable $grp
ipx::add_file myromtable.v $grp

set grp [ipx::add_file_group -type verilog:synthesis {} $ip]
set_property model_name myromtable $grp
ipx::add_file myromtable.v $grp

set grp [ipx::add_file_group -type misc {} $ip]
ipx::add_file romdata.hex $grp

ipx::import_top_level_hdl -top_module_name myromtable -top_level_hdl_file myromtable.v $ip
ipx::create_xgui_files $ip

ipx::check_integrity $ip
ipx::save_core $ip
 
