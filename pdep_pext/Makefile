
report.txt: report.py pdep_pext.rpt ice40.rpt ice40p.rpt ice40m.rpt
	python3 report.py > report.txt
	cat report.txt

pdep_pext.rpt: pdep_pext.v pdep_pext.ys
	yosys pdep_pext.ys

model: model.cc
	gcc -Wall -O0 -ggdb -o model model.cc

testbench: testbench.vvp
	vvp -N testbench.vvp

testbench_mc: testbench_mc.vvp
	vvp -N testbench_mc.vvp

testbench.vvp: pdep_pext.v testbench.v
	iverilog -o testbench.vvp pdep_pext.v testbench.v
	chmod -x testbench.vvp

testbench_mc.vvp: pdep_pext.v testbench.v
	iverilog -D PDEP_PEXT_MC -o testbench_mc.vvp pdep_pext.v testbench.v
	chmod -x testbench_mc.vvp

ice40.rpt: pdep_pext.v
	yosys -p 'synth_ice40 -top pdep_pext_syntop -blif ice40.blif' pdep_pext.v
	arachne-pnr -d 8k -o ice40.asc ice40.blif
	icetime -d hx8k -r ice40.rpt ice40.asc

ice40p.rpt: pdep_pext.v
	yosys -p 'read_verilog -D PDEP_PEXT_PIPELINE pdep_pext.v; synth_ice40 -top pdep_pext_syntop -blif ice40p.blif'
	arachne-pnr -d 8k -o ice40p.asc ice40p.blif
	icetime -d hx8k -r ice40p.rpt ice40p.asc

ice40m.rpt: pdep_pext.v
	yosys -p 'read_verilog pdep_pext.v; synth_ice40 -top pdep_pext_mc_syntop -blif ice40m.blif'
	arachne-pnr -d 8k -o ice40m.asc ice40m.blif
	icetime -d hx8k -r ice40m.rpt ice40m.asc

