create_project -force mytop1_prj
set_property part xc7z010clg225-2 [current_project]

set_property ip_repo_paths ../myip1 [current_fileset]
update_ip_catalog -rebuild

create_ip -vlnv example.com:examples:myip1:1.0 -module_name myip1_0
generate_target all [get_files myip1_0.xci]

add_files mytop1.v

create_ip_run [get_ips]
launch_runs [get_runs]
