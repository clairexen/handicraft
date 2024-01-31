read_verilog test.v
synth_design -top test -mode out_of_context -part xc7a15tcsg325-1
write_edif -force test_vivado_ooc.edn
close_design

read_verilog test.v
synth_design -top test -part xc7a15tcsg325-1
write_edif -force test_vivado.edn
close_design
