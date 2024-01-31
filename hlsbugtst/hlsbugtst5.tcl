# run: vivado_hls -f hlsbugtst5.tcl

open_project -reset hlsbugtst5
set_top hlsbugtst5
add_files hlsbugtst5.cc

open_solution -reset "solution"
set_part {xc7z045ffg676-1} -tool vivado
create_clock -period 10 -name default

csynth_design
exit
