# run with:
# /opt/Xilinx/Vivado_HLS/2017.1/bin/vivado_hls -f vivadobug04.tcl

open_project -reset vivadobug04_prj
set_top vivadobug04
add_files vivadobug04.cc
add_files -tb vivadobug04_tb.cc

open_solution -reset "solution"
set_part {xc7z045ffg676-1} -tool vivado
create_clock -period 10 -name default

csim_design
csynth_design
cosim_design

exit
