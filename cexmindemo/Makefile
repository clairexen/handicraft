define YOSYS_SCRIPT
echo on
read -sv top.sv
prep -flatten -top top
hierarchy -simcheck
write_rtlil design_rtl.il
check
opt -full
techmap
opt -fast
dffunmap
abc -g AND -fast
opt_clean
stat
write_rtlil design_aig.il
write_aiger -I -B -zinit -map design.aim design.aig
endef

export YOSYS_SCRIPT

all: trace.vcd trace_acm.vcd trace_acmx.vcd trace_init.vcd trace_rerun.vcd

design.aig: top.sv
	set -x; yosys -Qp "$$YOSYS_SCRIPT"

trace_%.aiw: design.aig
	yosys-abc -c 'read_aiger design.aig; strash; bmc3 -F 20 -v' \
	          -c 'write_cex -$(subst trace_,,$(subst .aiw,,$@)) $@'

trace_rerun.aiw: design.aig trace_ac.aiw
	yosys-abc -c 'read_aiger design.aig; read_cex trace_ac.aiw;' \
	          -c 'write_cex -ac $@'

trace.vcd: trace_ac.aiw
	yosys -QTp 'read_rtlil design_rtl.il' \
	      -p "sim -r $< -map design.aim -clock clock -vcd $@"

trace_%.vcd: trace_%.aiw
	yosys -QTp 'read_rtlil design_aig.il' \
	      -p "sim -r $< -map design.aim -clock clock -vcd $@"

trace.fst: trace.vcd
	vcd2fst trace.vcd trace.fst

trace_init.vcd: trace.fst
	yosys -QTp 'read_rtlil design_rtl.il' \
	      -p 'dump w:init w:state' \
	      -p 'sim -r trace.fst -scope top -w -vcd trace_init.vcd' \
	      -p 'dump w:init w:state'

print: trace_ac.aiw trace_rerun.aiw trace_acm.aiw trace_acmx.aiw trace_ancm.aiw trace_ancmx.aiw
	@echo ""
	@echo "          ============================================================"
	@echo ""
	paste trace_ac.aiw trace_rerun.aiw trace_acm.aiw trace_acmx.aiw | expand -t20
	@echo ""
	paste design.aim trace_ancm.aiw trace_ancmx.aiw | expand -t30
	@echo ""

show: trace.vcd trace_acmx.vcd
	twinwave trace.vcd trace.gtkw + trace_acmx.vcd trace.gtkw

clean:
	rm -f design*.il
	rm -f design*.aig
	rm -f design*.aim
	rm -f trace*.aiw
	rm -f trace*.vcd
	rm -f trace*.fst

.PHONY: all print show clean
