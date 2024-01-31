# run: vivado_hls -f hlsbugtst3a.tcl

open_project -reset hlsbugtst3a
set_top hlsbugtst3
add_files hlsbugtst3a.cc
add_files -tb hlsbugtst3_tb.cc
open_solution "solution1"
set_part {xc7z045ffg676-1}
create_clock -period 10 -name default
csim_design
csynth_design
cosim_design -trace_level all
exit
