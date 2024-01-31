N := 200
SIMOUT_TARGETS := $(addprefix mut_simout_,$(addsuffix .txt,$(shell seq 0 $(N))))
EQUIV_TARGETS := $(addprefix mut_equiv_,$(addsuffix /status,$(shell seq 0 $(N))))
SIM3_TARGETS := $(addprefix mut_sim3_,$(addsuffix /status,$(shell seq 0 $(N))))

all: $(SIMOUT_TARGETS) $(EQUIV_TARGETS) $(SIM3_TARGETS)

picorv32_nomut.il: picorv32.v
	yosys -o picorv32_nomut.il -p 'chparam -set REGS_INIT_ZERO 1 -set COMPRESSED_ISA 1 -set ENABLE_IRQ 1 -set ENABLE_IRQ_QREGS 0 -set BARREL_SHIFTER 1 picorv32; prep -top picorv32' picorv32.v

picorv32.mut: picorv32_nomut.il
	yosys -p 'mutate -list $(N) -ctrl mutsel 8 1 -o picorv32.mut' picorv32_nomut.il

picorv32_mut.il: picorv32_nomut.il picorv32.mut
	yosys -o picorv32_mut.il -s picorv32.mut picorv32_nomut.il

picorv32_mut.v: picorv32_mut.il
	yosys -b 'verilog -attr2comment' -o picorv32_mut.v picorv32_mut.il

testbench: testbench.v picorv32_mut.v
	iverilog -o testbench testbench.v picorv32_mut.v

mut_simout_%.txt: testbench
	vvp -N testbench +mut=$(subst .txt,,$(subst mut_simout_,,$@)) testbench > $@

mut_equiv_%/status: mut_equiv.sby picorv32_mut.il
	sby -f mut_equiv.sby $(subst /status,,$(subst mut_equiv_,,$@))

mut_sim3_%/status: mut_sim3.sby picorv32_mut.il
	sby -f mut_sim3.sby $(subst /status,,$(subst mut_sim3_,,$@))

zip:
	tar cvJf results.tar.xz mut_equiv_*/logfile.txt mut_equiv_*/status mut_sim3_*/logfile.txt mut_sim3_*/status mut_simout_*.txt picorv32.mut

analyze:
	bash analyze.sh

clean:
	rm -rf picorv32.mut picorv32_mut.il picorv32_mut.v picorv32_nomut.il
	rm -rf testbench testbench.vcd mut_simout_[0-9]*.txt mut_equiv_[0-9]*/ mut_sim3_[0-9]*/

.PHONY: all zip analyze clean
