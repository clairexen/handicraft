if {1} {
	create_project -part xc7a50tfgg484-1 -force vivadoprj vivadoprj
	read_verilog dumpwires.v
	read_xdc dumpwires.xdc
	launch_runs impl_1
	wait_on_run impl_1
	open_run impl_1

	write_checkpoint -force dumpwires.dcp
} else {
	open_checkpoint dumpwires.dcp
}


set X1 35
set Y1 42
set X2 46
set Y2 46

set tiles [get_tiles -filter "GRID_POINT_X>=$X1 && GRID_POINT_X<=$X2 && GRID_POINT_Y>=$Y1 && GRID_POINT_Y<=$Y2"]

set f [open "../../database/wires.txt" "w"]

foreach node [get_nodes -of_objects $tiles] {
  puts $f "node $node"
  foreach wire [get_wires -of_objects $node] {
    puts $f "  wire $wire"
  }
  puts $f ""
}

foreach pip [get_pips -of_objects $tiles] {
  puts $f "pip $pip"
  foreach node [get_nodes -uphill -of_objects $pip] {
    puts $f "  uphill $node"
  }
  foreach node [get_nodes -downhill -of_objects $pip] {
    puts $f "  downhill $node"
  }
  puts $f ""
}

close $f

