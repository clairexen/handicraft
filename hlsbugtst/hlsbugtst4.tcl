# run: vivado_hls -f hlsbugtst4.tcl

open_project -reset hlsbugtst4
set_top hlsbugtst4
add_files hlsbugtst4.cc
add_files -tb hlsbugtst4_tb.cc

open_solution -reset "solution"
set_part {xc7z045ffg676-1} -tool vivado
create_clock -period 10 -name default

csim_design
csynth_design
cosim_design -trace_level all
exit
