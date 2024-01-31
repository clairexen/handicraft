open_project -reset runhls_prj
set_top hls_uut
add_files hlsbugtst2.cc
add_files -tb testbench.cc

open_solution -reset "solution"
set_part {xc7z045ffg676-1} -tool vivado
create_clock -period 10 -name default
config_schedule -effort high -verbose
config_bind -effort medium

csim_design
csynth_design
cosim_design

exit
