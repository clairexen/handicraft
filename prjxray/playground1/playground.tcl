create_project -part xc7a50tfgg484-1 -force vivadoprj vivadoprj
read_verilog playground.v
read_xdc playground.xdc
launch_runs impl_1
wait_on_run impl_1
open_run impl_1

write_checkpoint -force playground.dcp

set f [open "test000.tags" "w"]
puts $f [join [lsort -unique [get_pips -of_objects [get_nets -hierarchical]]] \n]
close $f

write_bitstream -force test000.bit

for {set test_idx 1} {$test_idx < 500} {incr test_idx} {
	for {set lut_idx 0} {$lut_idx < 8} {incr lut_idx} {
		set lut [get_cell "playground/LUTS\[${lut_idx}\]"]
		set bel_pins [list A1 A2 A3 A4 A5 A6]
		set lock_pins ""

		for {set pin_idx 0} {$pin_idx < 6} {incr pin_idx} {
			set bel_pin_idx [expr int([llength $bel_pins]*rand())]
			set lock_pins "${lock_pins} I${pin_idx}:[lindex $bel_pins $bel_pin_idx]"
			set bel_pins [lreplace $bel_pins $bel_pin_idx $bel_pin_idx]
		}

		reset_property LOCK_PINS $lut
		set_property LOCK_PINS $lock_pins $lut
	}

	route_design

	set f [open [format "test%03d.tags" $test_idx] "w"]
	puts $f [join [lsort -unique [get_pips -of_objects [get_nets -hierarchical]]] \n]
	close $f

	write_bitstream -force [format "test%03d.bit" $test_idx]
}

