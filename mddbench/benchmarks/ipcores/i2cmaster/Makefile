help:
	@echo "This makefile is used to generate the static benchmark files"
	@echo "that are checked into the mddbench source repository."

static: trace1.fst trace2.fst

RTL_TOP := i2c_master_top
SIM_TOP := tst_bench_top
SIM_UUT_1 := tst_bench_top.i2c_top
SIM_UUT_2 := tst_bench_top.i2c_top2

SV_INCLUDES := rtl/timescale.v
SV_INCLUDES += rtl/i2c_master_defines.v

SV_RTL_SRC := rtl/i2c_master_bit_ctrl.v
SV_RTL_SRC += rtl/i2c_master_byte_ctrl.v
SV_RTL_SRC += rtl/i2c_master_top.v

SV_SIM_SRC := sim/i2c_slave_model.v
SV_SIM_SRC += sim/spi_slave_model.v
SV_SIM_SRC += sim/tst_bench_top.v
SV_SIM_SRC += sim/wb_master_model.v

simtrace.fst: $(SV_INCLUDES) $(SV_RTL_SRC) $(SV_SIM_SRC)
	iverilog -o testbench -s $(SIM_TOP) -Irtl $(SV_RTL_SRC) $(SV_SIM_SRC)
	vvp -N ./testbench -fst

design.sv: $(SV_INCLUDES) $(SV_RTL_SRC)
	grep -hv '^`include' $(SV_INCLUDES) $(SV_RTL_SRC) > design.sv

trace1.fst: simtrace.fst design.sv
	yosys -p 'read -sv design.sv; prep -top $(RTL_TOP); sim -q -sim -scope $(SIM_UUT_1) -clock wb_clk_i -r simtrace.fst -fst trace1.fst'

trace2.fst: simtrace.fst design.sv
	yosys -p 'read -sv design.sv; prep -top $(RTL_TOP); sim -q -sim -scope $(SIM_UUT_2) -clock wb_clk_i -r simtrace.fst -fst trace2.fst'

mrproper: clean
	rm -f design.sv trace1.fst trace2.fst

clean:
	rm -f testbench simtrace.fst

.PHONY: help static mrproper clean
