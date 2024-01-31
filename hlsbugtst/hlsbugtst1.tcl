# run: vivado_hls -f hlsbugtst1.tcl

open_project hlsbugtst1
set_top hlsbugtst1
add_files hlsbugtst1.cc

open_solution -reset "solution"
set_part {xc7z045ffg676-1} -tool vivado
create_clock -period 10 -name default
config_bind -effort high

csynth_design
export_design -display_name hlsbugtst1 -evaluate verilog \
              -format ip_catalog -vendor myvendor -version 0.0.0

exit
