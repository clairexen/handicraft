show: run_yosys.log
	grep -H 'Number of cells:' yosys_*_*.txt | sed -r 's,(.txt:.*: *|_),\t,g'

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
