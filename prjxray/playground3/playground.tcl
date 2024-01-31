
create_project -force -part xc7a50tfgg484-1 synth
read_xdc playground.xdc
read_verilog playground.v
read_verilog testbox_0.v
read_verilog testbox_1.v
read_verilog testbox_A.v
read_verilog testbox_B.v

# Build testboxes

synth_design -mode out_of_context -flatten_hierarchy rebuilt -top testbox_0
write_checkpoint -force testbox_0.dcp

synth_design -mode out_of_context -flatten_hierarchy rebuilt -top testbox_1
write_checkpoint -force testbox_1.dcp

synth_design -mode out_of_context -flatten_hierarchy rebuilt -top testbox_A
write_checkpoint -force testbox_A.dcp

synth_design -mode out_of_context -flatten_hierarchy rebuilt -top testbox_B
write_checkpoint -force testbox_B.dcp

# Build static part

synth_design -top top

create_pblock test_pblock
resize_pblock test_pblock -add SLICE_X20Y135:SLICE_X23Y135

add_cells_to_pblock test_pblock [get_cells box]
set_property HD.RECONFIGURABLE TRUE [get_cells box]

read_checkpoint -cell box testbox_0.dcp

opt_design
place_design
route_design

lock_design -level routing
write_checkpoint -force static.dcp

