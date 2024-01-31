
## Build sequencer1 and sequencer2

create_project -force -part xc7a35tcpg236-1 synth
read_verilog top.v
read_verilog sequencer1.v
read_verilog sequencer2.v

synth_design -mode out_of_context -flatten_hierarchy rebuilt -top sequencer1
write_checkpoint -force sequencer1.dcp

synth_design -mode out_of_context -flatten_hierarchy rebuilt -top sequencer2
write_checkpoint -force sequencer2.dcp
close_project

## Build top design

create_project -force -part xc7a35tcpg236-1 synth
read_verilog top.v
read_xdc basys3.xdc

synth_design -top top

create_pblock sequencer_pblock
add_cells_to_pblock sequencer_pblock [get_cells seq_inst]
resize_pblock sequencer_pblock -add SLICE_X0Y9:SLICE_X3Y10
set_property HD.RECONFIGURABLE TRUE [get_cells seq_inst]

## Link sequencer1 and implement

read_checkpoint -cell seq_inst sequencer1.dcp

opt_design
place_design
phys_opt_design
route_design

lock_design -level routing

write_checkpoint -force sequencer1_routed.dcp
write_checkpoint -force -cell seq_inst sequencer1_routed_rm.dcp

## Also write a design check point with only the static part

update_design -cell seq_inst -black_box
lock_design -level routing
write_checkpoint -force static_routed.dcp
close_project

## Link sequencer2 and implement
# (we reload the design here, so this could be in a different script)

open_checkpoint static_routed.dcp
read_checkpoint -cell seq_inst sequencer2.dcp

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force sequencer2_routed.dcp
write_checkpoint -force -cell seq_inst sequencer2_routed_rm.dcp

## Verify design check-points

pr_verify -full_check sequencer1_routed.dcp sequencer2_routed.dcp -file pr_verify_seq1_seq2.log
pr_verify -full_check -initial static_routed.dcp -additional {sequencer1_routed.dcp sequencer2_routed.dcp} -file pr_verify_all.log
close_project

## Write bitstream files

open_checkpoint sequencer1_routed.dcp
write_bitstream -force sequencer1
close_project

open_checkpoint sequencer2_routed.dcp
write_bitstream -force sequencer2
close_project

