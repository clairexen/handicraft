show: run_yosys.log vivado.rpt
	grep -H 'Number of cells:' yosys_*_*.txt | sed -r 's,(.txt:.*: *|_),\t,g'; echo
	grep --color=never -B1 -A8 'Total LUTs' vivado.rpt
	grep -A 24 'Report Cell Usage:' vivado.log; echo
	grep 8-3886 vivado.log && echo "$$(grep -c 8-3886 vivado.log) lines of [Synth 8-3886]"; echo

run_yosys.log: run_yosys.ys *.v 
	yosys -v3 -l run_yosys.log run_yosys.ys

vivado.rpt: vivado.sh vivado.tcl vivado.xdc
	bash vivado.sh vivado -mode batch -source vivado.tcl -log vivado.log -nojournal

clean:
	rm -f run_yosys.log
	rm -f yosys_*_*.txt
	rm -f vivado*.jou
	rm -f vivado*.log
	rm -f vivado.rpt

.PHONY: show clean
