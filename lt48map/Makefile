SHELL := bash

run: picorv32 nerv test
	python3 summary.py | tee results.txt

picorv32: picorv32_l4.stat picorv32_lt4.stat
nerv: nerv_l4.stat nerv_lt4.stat
test: test_l4.stat test_lt4.stat

cells.genlib: Makefile makecells.py
	python3 makecells.py > cells.genlib

picorv32_l4.stat: Makefile picorv32.v
	yosys -o picorv32_l4.il -l picorv32_l4.log picorv32.v \
		-p 'synth -flatten -top picorv32; abc -lut 4' \
		-p 'tee -o picorv32_l4.stat stat'

picorv32_lt4.stat: Makefile cells.genlib picorv32.v
	yosys -o picorv32_lt4.il -l picorv32_lt4.log picorv32.v \
		-p 'synth -flatten -top picorv32; abc -genlib cells.genlib' \
		-p 'tee -o picorv32_lt4.stat stat'

nerv_l4.stat: Makefile nerv.sv
	yosys -o nerv_l4.il -l nerv_l4.log nerv.sv \
		-p 'synth -flatten -top nerv; abc -lut 4' \
		-p 'tee -o nerv_l4.stat stat'

nerv_lt4.stat: Makefile cells.genlib nerv.sv
	yosys -o nerv_lt4.il -l nerv_lt4.log nerv.sv \
		-p 'synth -flatten -top nerv; abc -genlib cells.genlib' \
		-p 'tee -o nerv_lt4.stat stat'

test_l4.stat: Makefile test.sv
	yosys -o test_l4.il -l test_l4.log test.sv \
		-p 'synth -flatten -top test; abc -lut 4' \
		-p 'tee -o test_l4.stat stat'

test_lt4.stat: Makefile cells.genlib test.sv
	yosys -o test_lt4.il -l test_lt4.log test.sv \
		-p 'synth -flatten -top test; abc -genlib cells.genlib' \
		-p 'tee -o test_lt4.stat stat'

clean:
	rm -f picorv32_{l4,lt4}.{log,il,stat}
	rm -f nerv_{l4,lt4}.{log,il,stat}
	rm -f test_{l4,lt4}.{log,il,stat}
