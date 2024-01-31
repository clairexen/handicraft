# run: vivado_hls -f hlsbugtst3.tcl

open_project -reset hlsbugtst3
set_top hlsbugtst3
add_files hlsbugtst3.cc
add_files -tb hlsbugtst3_tb.cc
open_solution "solution1"
set_part {xc7z045ffg676-1}
create_clock -period 10 -name default
csim_design
csynth_design
cosim_design -trace_level all
exit
