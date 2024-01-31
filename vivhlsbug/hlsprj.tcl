open_project hlsprj
set_top demo
add_files demo.cc
add_files demo.h
add_files -tb demo_tb.cc

open_solution -reset "solution"
set_part {xc7z045ffg676-1} -tool vivado
create_clock -period 10 -name default

csim_design
csynth_design

# exec sed -i "/^demo_func2/,/;/ s/.ap_start(.*)/.ap_start(1'b1)/;" hlsprj/solution/syn/verilog/demo.v
# cosim_design

exit
