create_project -force mytop2_prj
set_property part xc7z010clg225-2 [current_project]

set_property ip_repo_paths {../myip1 ../myip2} [current_fileset]
update_ip_catalog -rebuild

create_ip -vlnv example.com:examples:myip2:1.0 -module_name myip2_0
generate_target all [get_files myip2_0.xci]

add_files mytop2.v

create_ip_run [get_ips]
launch_runs [get_runs]
