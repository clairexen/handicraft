
all: dcmctrl.rpt testbench.vcd

testbench.vcd: testbench
	./testbench

testbench: testbench.v dcmctrl.v
	iverilog -o testbench testbench.v dcmctrl.v

dcmctrl.blif: dcmctrl.v
	yosys -p 'synth_ice40 -top dcmctrl -blif dcmctrl.blif' dcmctrl.v

dcmctrl.asc: dcmctrl.blif
	arachne-pnr -o dcmctrl.asc dcmctrl.blif

dcmctrl.rpt: dcmctrl.asc
	icetime -d hx1k -r dcmctrl.rpt dcmctrl.asc

dcmctrl_syn.v: dcmctrl.blif
	yosys -p 'read_blif -wideports dcmctrl.blif; write_verilog dcmctrl_syn.v'

testbench_syn: testbench.v dcmctrl_syn.v
	iverilog -o testbench_syn testbench.v dcmctrl_syn.v `yosys-config --datdir/ice40/cells_sim.v`

test_device.blif: test_device.v dcmctrl.v
	yosys -p 'synth_ice40 -top top -blif test_device.blif' test_device.v dcmctrl.v

test_device.asc: test_device.pcf test_device.blif
	arachne-pnr -d 1k -o test_device.asc -p test_device.pcf test_device.blif

test_device.bin: test_device.asc
	icetime -d hx1k -c 12 -r test_device.rpt test_device.asc
	icepack test_device.asc test_device.bin

clean:
	rm -f testbench.vcd testbench
	rm -f dcmctrl.asc dcmctrl.blif dcmctrl.rpt
	rm -f test_device.asc test_device.blif test_device.rpt test_device.bin
	rm -f dcmctrl_syn.v testbench_syn

