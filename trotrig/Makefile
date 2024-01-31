
test: testbench.exe
	vvp -N testbench.exe

testbench.exe: testbench.v trotrig.v
	iverilog -s testbench -o $@ $^
	chmod -x $@

clean:
	rm -f testbench.exe testbench.vcd

