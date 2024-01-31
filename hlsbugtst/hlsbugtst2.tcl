# run: vivado_hls -f hlsbugtst2.tcl

open_project -reset hlsbugtst2
set_top racalc_mta_stage2
add_files hlsbugtst2.cc

open_solution -reset "solution"
set_part {xc7z045ffg676-1} -tool vivado
create_clock -period 10 -name default
config_bind -effort high

csynth_design
exit
