create_project -force -part xc7a50tfgg484-1 synth
read_xdc playground.xdc
read_verilog playground.v

synth_design -top top
opt_design
place_design

write_checkpoint -force playground.dcp

set_property LOCK_PINS {I0:A1 I1:A2 I2:A3 I3:A4 I4:A5 I5:A6} [get_cells mylut]
set_property -dict {IS_LOC_FIXED 1 IS_BEL_FIXED 1 BEL SLICEL.A6LUT} [get_cells mylut]

foreach tile [get_tiles -filter {(TYPE =~ CLBL?_?) && ((NAME =~ "*Y0") || (NAME =~ "*Y50") || NAME =~ "*Y100")}] {
	set REGION [get_clock_regions -of_objects $tile]
	set COLUMN [get_property COLUMN $tile]
	set ROW [get_property ROW $tile]

	puts ""
	puts ""
	puts ""
	puts ""
	puts ""
	puts "============================================================"
	puts "==== $tile: $REGION $ROW $COLUMN"
	puts ""

	set_property LOC [lindex [get_sites -of_objects $tile] 0] [get_cells mylut]
	route_design

	set_property INIT "64'h8000000000000000" [get_cells mylut]
	write_bitstream -force data_${REGION}_${COLUMN}_${ROW}_${tile}_A.bit

	set_property INIT "64'h8000000000000001" [get_cells mylut]
	write_bitstream -force data_${REGION}_${COLUMN}_${ROW}_${tile}_B.bit
}
