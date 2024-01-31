CC = clang
CXX = clang
CXXFLAGS = -std=c++11 -Wall -Wextra -ggdb
LDFLAGS = -ggdb
LDLIBS = -lstdc++

demo: demo.dat testbench.vvp
	vvp -N testbench.vvp +vcd +meminit=demo.dat

test: $(addsuffix .ok,$(basename $(wildcard tests/*.ios)))
.SECONDARY: $(addsuffix .out,$(basename $(wildcard tests/*.ios)))
.SECONDARY: $(addsuffix .dat,$(basename $(wildcard tests/*.ios)))

tests/%.ok: tests/%.out
	python3 testbench.py $(basename $<).ios $<
	touch $@

tests/%.out: tests/%.dat testbench.vvp
	vvp -N testbench.vvp +meminit=$< | tee $@.new
	mv $@.new $@

tests/%.dat: tests/%.ios ioshim_asm
	./ioshim_asm < $< > $@.new
	mv $@.new $@

report: report.txt
	echo; sed -re '/^=|Chip area/ ! d; /^=/ ! { p; s/.*//; }' report.txt

timer:
	OpenTimer < timer.conf

ioshim_asm: ioshim_asm.o

demo.dat: demo.ios ioshim_asm
	./ioshim_asm < demo.ios > demo.dat.new
	mv demo.dat.new demo.dat

testbench.vvp: testbench.v ioshim_cpu.v ioshim_mem.v ioshim_regs.v ioshim_alu.v
	iverilog -o $@ $^
	chmod -x testbench.vvp

report.txt: ioshim_*.v vsclib013.lib report.ys
	yosys report.ys
	mv report.new report.txt

vsclib013.lib:
	wget -O vsclib013.lib.new http://www.vlsitechnology.org/synopsys/vsclib013.lib
	mv vsclib013.lib.new vsclib013.lib

clean:
	rm -f *.o ioshim_asm vsclib013.lib
	rm -f demo.dat report.txt timer.v timer.abc
	rm -f testbench.vvp testbench.vcd
	rm -f tests/*.dat tests/*.ok tests/*.out

.PHONY: demo test report timer clean

