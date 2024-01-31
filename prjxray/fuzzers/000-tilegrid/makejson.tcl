create_project -part $env(XRAY_PART) -force vivadoprj vivadoprj
read_verilog dummy.v
synth_design -top dummy

set f [open "../../database/tilegrid.json" "w"]

puts $f "\{"
set first_entry 1

set grid_x1 [lindex $env(XRAY_REGION) 0]
set grid_y1 [lindex $env(XRAY_REGION) 1]
set grid_x2 [lindex $env(XRAY_REGION) 2]
set grid_y2 [lindex $env(XRAY_REGION) 3]

foreach tile [get_tiles] {
	if {$first_entry != 1} {
		puts $f "  \},"
	}
	puts $f "  \"${tile}\": \{"
	set first_entry 0

	set x [get_property GRID_POINT_X $tile]
	set y [get_property GRID_POINT_Y $tile]

	if {$grid_x1 <= $x && $x <= $grid_x2 && $grid_y1 <= $y && $y <= $grid_y2} {
		puts $f "    \"FUZZ\": \"1\","
	} else {
		puts $f "    \"FUZZ\": \"0\","
	}

	set plist [list_property $tile]
	set plist [concat \
			[lsearch -inline $plist NAME] \
			[lsearch -inline $plist CLASS] \
			[lsearch -inline $plist CLOCK_REGION] \
			[lsearch -inline $plist GRID_POINT_X] \
			[lsearch -inline $plist GRID_POINT_Y] \
			[lsearch -inline $plist TILE_TYPE] \
			[lsearch -inline $plist SITE_TYPE]]

	foreach p $plist {
		set v [get_property $p $tile]
		puts $f "    \"${p}\": \"${v}\","
	}

	puts $f "    \"SITES\": \{"

	if {[get_property NUM_SITES $tile] != 0} {
		set slist [get_sites -of_objects $tile]
		foreach s $slist {
			puts $f "      \"${s}\": \{"

			set plist [list_property $s]
			set plist [concat \
					[lsearch -inline $plist NAME] \
					[lsearch -inline $plist CLASS] \
					[lsearch -inline $plist CLOCK_REGION] \
					[lsearch -inline $plist GRID_POINT_X] \
					[lsearch -inline $plist GRID_POINT_Y] \
					[lsearch -inline $plist TILE_TYPE] \
					[lsearch -inline $plist SITE_TYPE]]

			foreach p $plist {
				set v [get_property $p $s]
				if {$p != [lindex $plist end]} {
					puts $f "        \"${p}\": \"${v}\","
				} else {
					puts $f "        \"${p}\": \"${v}\""
				}
			}

			if {$s != [lindex $slist end]} {
				puts $f "      \},"
			} else {
				puts $f "      \}"
			}
		}
	}

	puts $f "    \}"
}

puts $f "  \}"
puts $f "\}"

close $f
