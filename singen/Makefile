
all: singen.bin singen.rpt

singen.bin: singen.asc
	icepack singen.asc singen.bin

singen.rpt: singen.asc
	icetime -tmd hx1k -r singen.rpt singen.asc

singen.asc: singen.blif singen.pcf
	arachne-pnr -d 1k -P tq144 -p singen.pcf -o singen.asc singen.blif

singen.blif: singen.v sintable.hex
	yosys -v2 -l singen.log -p 'synth_ice40 -blif singen.blif' singen.v

sintable.hex: sintable.py
	python3 sintable.py

testbench.exe: testbench.v singen.v sintable.hex
	iverilog -o testbench.exe testbench.v singen.v
	chmod -x testbench.exe

test: testbench.exe
	vvp -N testbench.exe

prog: singen.bin
	iceprog -S singen.bin

clean:
	rm -f singen.asc singen.bin singen.blif singen.log singen.rpt
	rm -f singen.log testbench.exe testbench.vcd sintable.hex

.PHONY: all test prog clean

