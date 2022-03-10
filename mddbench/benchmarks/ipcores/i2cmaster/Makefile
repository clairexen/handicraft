help:
	@echo "This makefile is used to generate the static benchmark files"
	@echo "that are checked into the mddbench source repository."

static: trace.fst

VLSRC := rtl/i2c_master_bit_ctrl.v
VLSRC += rtl/i2c_master_byte_ctrl.v
VLSRC += rtl/i2c_master_top.v
VLSRC += sim/i2c_slave_model.v
VLSRC += sim/spi_slave_model.v
VLSRC += sim/tst_bench_top.v
VLSRC += sim/wb_master_model.v

trace.fst: $(VLSRC) rtl/timescale.v rtl/i2c_master_defines.v
	iverilog -o testbench -s tst_bench_top -Irtl $(VLSRC)
	vvp -N ./testbench
	vcd2fst trace.vcd trace.fst

mrproper: clean
	rm -f trace.fst

clean:
	rm -f testbench trace.vcd

.PHONY: help static mrproper clean
