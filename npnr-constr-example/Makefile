default: ringosc.asc

ringosc.json: ringosc.ys ringosc.v
	yosys ringosc.ys

ringosc.asc: ringosc.json ringosc.pcf ringosc.py
	nextpnr-ice40 --json ringosc.json --pcf ringosc.pcf --pre-place ringosc.py --hx1k --package tq144
