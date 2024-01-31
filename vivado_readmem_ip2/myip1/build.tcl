create_project -ip -force myip1_prj
ipx::create_core example.com examples myip1 1.0

set_property description {Example IP for $readmemh() usage in IP HDL} [ipx::current_core]
set_property display_name myip1 [ipx::current_core]
set_property root_directory . [ipx::current_core]
set_property taxonomy {{/Example}} [ipx::current_core]

set_property supported_families {
	zynq	Production
} [ipx::current_core]

set grp [ipx::add_file_group -type verilog:simulation {} [ipx::current_core]]
set_property model_name myip1 $grp
ipx::add_file myip1.v $grp

set grp [ipx::add_file_group -type verilog:synthesis {} [ipx::current_core]]
set_property model_name myip1 $grp
ipx::add_file myip1.v $grp

set grp [ipx::add_file_group -type misc {} [ipx::current_core]]
ipx::add_file myip1.dat $grp

ipx::import_top_level_hdl -top_module_name myip1 -top_level_hdl_file myip1.v [ipx::current_core]

ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]
ipx::save_core [ipx::current_core]
