
all: cmpnets.log

logic6502.v: netlist.v sw2sr.sh sw2sr.ys sw2sr.cc
	bash sw2sr.sh

switchsim: netlist.c switchsim.cc
	clang -o switchsim -std=c++11 switchsim.cc -lstdc++ -O0 -ggdb

cmpnets.v: switchsim logic6502.v
	./switchsim

cmpnets.log: cmpnets.v testbench.v logic6502.v
	iverilog -o testbench testbench.v logic6502.v
	./testbench | tee cmpnets.log

inspect:
	yosys -b "" -p "read_ilang logic6502.il; read_verilog nmos.v; select -module MOS6502; shell"

clean:
	rm -f logic6502.v testbench cmpnets.txt cmpnets.log testbench.vcd
	rm -f sw2sr.so sw2sr.log sw2sr.d switchsim cmpnets.v logic6502.il

.PHONY: all inspect clean

