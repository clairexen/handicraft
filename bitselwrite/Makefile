all: equiv reference.log improved.log improved2.log

equiv: equiv.sby equiv.sv improved.v improved2.log
	sby -f equiv.sby

reference.log: reference.v
	yosys -l reference.log -p synth_xilinx reference.v

improved.log: improved.v
	yosys -l improved.log -p synth_xilinx improved.v

improved2.log: improved2.v
	yosys -l improved2.log -p synth_xilinx improved2.v

clean:
	rm -rf equiv reference.log improved.log improved2.log
