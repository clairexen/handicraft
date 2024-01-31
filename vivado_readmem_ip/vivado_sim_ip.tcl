# vivado -mode batch -nolog -nojournal -source vivado_sim_ip.tcl

create_project -force testprj_sim
set_property part xc7z030ffg676-1 [current_project]
set_property ip_repo_paths [pwd]/ip_myromtable [current_fileset]
update_ip_catalog -rebuild

create_ip -vlnv example.com:example_cores:myromtable:* -module_name romtab
add_files top.v

# workaround for simulating $readmem("romdata.hex")
#add_files -fileset [current_fileset -simset] ip_myromtable/romdata.hex

set_property top top [current_fileset -simset]
generate_target all [get_ips]
launch_simulation

