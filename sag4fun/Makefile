all: cmos32c.txt cmos32s.txt run32

run32: sag4fun32c sag4fun32s
	./sag4fun32c
	./sag4fun32s

sag4fun32c: testbench.v sag4fun.v
	iverilog -D TYPE_32 -D TYPE_C -o $@ $^

sag4fun32s: testbench.v sag4fun.v
	iverilog -D TYPE_32 -D TYPE_S -o $@ $^

cmos32c.txt: sag4fun.v
	yosys -p 'synth -flatten -top SAG4Fun32C; abc -g cmos; opt -fast' \
		-p 'tee -o cmos32c.txt stat -tech cmos; tee -a cmos32c.txt ltp -noff' sag4fun.v

cmos32s.txt: sag4fun.v
	yosys -p 'synth -flatten -top SAG4Fun32S; abc -g cmos; opt -fast' \
		-p 'tee -o cmos32s.txt stat -tech cmos; tee -a cmos32s.txt ltp -noff' sag4fun.v

clean:
	rm -f cmos32c.txt cmos32s.txt sag4fun32c sag4fun32s
