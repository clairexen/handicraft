# vivado -mode batch -nolog -nojournal -source vivado_synth_ip.tcl

create_project -force testprj_synth
set_property part xc7z030ffg676-1 [current_project]
set_property ip_repo_paths [pwd]/ip_myromtable [current_fileset]
update_ip_catalog -rebuild

create_ip -vlnv example.com:example_cores:myromtable:* -module_name romtab
add_files top.v

set_property top top [current_fileset]
generate_target all [get_ips]
synth_ip [get_ips]
synth_design

