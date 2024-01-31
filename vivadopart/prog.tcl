
# initialize programmer

open_hw
connect_hw_server
open_hw_target [lindex [get_hw_targets] 0]
set hwdev [lindex [get_hw_devices] 0]

# program sequencer1.bit

set_property PROGRAM.FILE sequencer1.bit $hwdev
program_hw_devices $hwdev

# functions for switching sequencer

proc sequencer1 {} {
	puts "Switching to sequencer1.."
	set hwdev [lindex [get_hw_devices] 0]
	set_property PROGRAM.FILE sequencer1_sequencer_pblock_partial.bit $hwdev
	program_hw_devices $hwdev
}

proc sequencer2 {} {
	puts "Switching to sequencer2.."
	set hwdev [lindex [get_hw_devices] 0]
	set_property PROGRAM.FILE sequencer2_sequencer_pblock_partial.bit $hwdev
	program_hw_devices $hwdev
}

# switch between sequencers in a loop

while 1 {
	sequencer1
	after 3000
	sequencer2
	after 3000
}

