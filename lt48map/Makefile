run: picorv nerv test

picorv32: picorv32_l4.log picorv32_lt4.log
nerv: nerv_l4.log nerv_lt4.log
test: test_l4.log test_lt4.log

cells.genlib: Makefile makecells.py
	python3 makecells.py > cells.genlib

picorv32_l4.log: picorv32_l4.il
picorv32_l4.il: Makefile picorv32.v
	yosys -o picorv32_l4.il -l picorv32_l4.log picorv32.v \
		-p 'synth -flatten -top picorv32; abc -lut 4; stat'

picorv32_lt4.log: picorv32_lt4.il
picorv32_lt4.il: Makefile cells.genlib picorv32.v
	yosys -o picorv32_lt4.il -l picorv32_lt4.log picorv32.v \
		-p 'synth -flatten -top picorv32; abc -genlib cells.genlib; stat'

nerv_l4.log: nerv_l4.il
nerv_l4.il: Makefile nerv.sv
	yosys -o nerv_l4.il -l nerv_l4.log nerv.sv \
		-p 'synth -flatten -top nerv; abc -lut 4; stat'

nerv_lt4.log: nerv_lt4.il
nerv_lt4.il: Makefile cells.genlib nerv.sv
	yosys -o nerv_lt4.il -l nerv_lt4.log nerv.sv \
		-p 'synth -flatten -top nerv; abc -genlib cells.genlib; stat'

test_l4.log: test_l4.il
test_l4.il: Makefile test.sv
	yosys -o test_l4.il -l test_l4.log test.sv \
		-p 'synth -flatten -top test; abc -lut 4; stat'

test_lt4.log: test_lt4.il
test_lt4.il: Makefile cells.genlib test.sv
	yosys -o test_lt4.il -l test_lt4.log test.sv \
		-p 'synth -flatten -top test; abc -genlib cells.genlib; stat'

clean:
	rm -f picorv32_l4.log picorv32_l4.il picorv32_lt4.log picorv32_lt4.il
	rm -f nerv_l4.log nerv_l4.il nerv_lt4.log nerv_lt4.il
	rm -f test_l4.log test_l4.il test_lt4.log test_lt4.il
