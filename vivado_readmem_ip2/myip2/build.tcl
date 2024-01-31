create_project -force myip2_prj
set_property part xc7z010clg225-2 [current_project]

set_property ip_repo_paths ../myip1 [current_fileset]
update_ip_catalog -rebuild

create_ip -vlnv example.com:examples:myip1:1.0 -module_name myip1_0
generate_target all [get_files myip1_0.xci]

add_files myip2.v

# This would be working fine:
#   create_ip_run [get_ips myip1_0]
#   launch_runs [get_runs]
#
# ..produces the log message:
# [Synth 8-3876] $readmem data file 'myip1.dat' is read successfully ["/.../myip2/myip2_prj.srcs/sources_1/ip/myip1_0/myip1.v":3]

ipx::package_project -root_dir . -vendor example.com -library examples
set_property name myip2 [ipx::current_core]
set_property version 1.0 [ipx::current_core]

set_property description {Example IP for IP usage in IP} [ipx::current_core]
set_property display_name myip2 [ipx::current_core]
set_property root_directory . [ipx::current_core]
set_property taxonomy {{/Example}} [ipx::current_core]

set_property supported_families {
	zynq Production
} [ipx::current_core]

ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]
ipx::save_core [ipx::current_core]
