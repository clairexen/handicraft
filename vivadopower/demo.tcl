create_project -part xc7a35tcpg236-1 -force vivadoprj

read_verilog demo.v
read_verilog testbench.v
read_xdc demo.xdc

set_property top demo [current_fileset]

launch_runs synth_1
wait_on_run synth_1


## launch_simulation
## source testbench.tcl
## run 100 us
## launch_runs impl_1
## open_run synth_1 -name synth_1
## launch_simulation -mode post-synthesis -type timing
## source testbench.tcl
## run 100 us
## restart
## open_saif -help
## open_saif demo.saif
## log_saif -help
## log_saif [ get_objects ]
## run 100 us
## close_saif
## read_saif {/home/clifford/Work/handicraft/2016/vivadopower/vivadoprj.sim/sim_1/synth/timing/demo.saif}
## report_power -name {power_1}
## history

