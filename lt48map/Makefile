run: picorv32_l4.log nerv_l4.log picorv32_lt4.log nerv_lt4.log

picorv32_l4.log: picorv32_l4.il
picorv32_l4.il: Makefile picorv32.v
	yosys -o picorv32_l4.il -l picorv32_l4.log picorv32.v \
		-p 'synth -flatten -top picorv32; abc -lut 4; stat'

nerv_l4.log: nerv_l4.il
nerv_l4.il: Makefile lt4map.genlib nerv.sv
	yosys -o nerv_l4.il -l nerv_l4.log nerv.sv \
		-p 'synth -flatten -top nerv; abc -lut 4; stat'

picorv32_lt4.log: picorv32_lt4.il
picorv32_lt4.il: Makefile lt4map.genlib picorv32.v
	yosys -o picorv32_lt4.il -l picorv32_lt4.log picorv32.v \
		-p 'synth -flatten -top picorv32; abc -genlib lt4map.genlib; stat'

nerv_lt4.log: nerv_lt4.il
nerv_lt4.il: Makefile lt4map.genlib nerv.sv
	yosys -o nerv_lt4.il -l nerv_lt4.log nerv.sv \
		-p 'synth -flatten -top nerv; abc -genlib lt4map.genlib; stat'

clean:
	rm -f picorv32_l4.log nerv_l4.log picorv32_l4.il nerv_l4.il
	rm -f picorv32_lt4.log nerv_lt4.log picorv32_lt4.il nerv_lt4.il
