if {0} {
	create_project -part xc7a50tfgg484-1 -force vivadoprj vivadoprj
	read_verilog playground.v
	read_xdc playground.xdc
	launch_runs impl_1
	wait_on_run impl_1
	open_run impl_1

	write_checkpoint -force playground.dcp
} else {
	open_checkpoint playground.dcp
}

for {set y1 1} {$y1 < 10} {incr y1} {
for {set y2 1} {$y2 < 10} {incr y2} {
for {set mode 1} {$mode <= 3} {incr mode} {
	set_property IS_ROUTE_FIXED 0 [get_nets q1]
	route_design -unroute -nets [get_nets q1]

	set n1 [get_nodes -of_objects [get_site_pins -filter {DIRECTION == "OUT"} -of_objects [get_nets q1]]]
	set n2 [get_nodes INT_L_X2Y11${y1}/EE2BEG1]

	if {$mode == 1} {
		set n3 [get_nodes INT_R_X15Y134/NR1BEG0]
		set n4 [get_nodes INT_R_X15Y135/NR1BEG0]
	} elseif {$mode == 2} {
		set n3 [get_nodes INT_L_X14Y134/NE2BEG0]
		set n4 [get_nodes INT_R_X15Y135/NR1BEG0]
	} else {
		set n3 [get_nodes INT_L_X14Y104/NE2BEG0]
		set n4 [get_nodes INT_R_X15Y105/NR1BEG0]
	}

	set n5 [get_nodes INT_L_X4Y11${y2}/EE2BEG2]
	set n6 [get_nodes -of_objects [get_site_pins -filter {DIRECTION == "IN"} -of_objects [get_nets q1]]]

	set p1 [find_routing_path -from $n1 -to $n2]
	set p2 [lreplace [find_routing_path -from $n2 -to $n3] 0 0]
	set p3 [find_routing_path -from $n4 -to $n5]
	set p4 [lreplace [find_routing_path -from $n5 -to $n6] 0 0]

	set_property FIXED_ROUTE "$p1 $p2 $p3 $p4" [get_nets q1]

	set f [open test${y1}${y2}${mode}.route "w"]
	puts $f [join [concat $p1 $p2 $p3 $p4] "\n"]
	close $f

	set f [open test${y1}${y2}${mode}.tags "w"]
	puts $f [get_pips -of_objects [get_nets q1] -filter {NAME == "INT_R_X15Y135/INT_R.NE2END0->>NR1BEG0" || NAME == "INT_R_X15Y135/INT_R.NR1END0->>NR1BEG0"}]
	close $f

	write_bitstream -force test${y1}${y2}${mode}.bit
}}}
