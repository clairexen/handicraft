
ice40sim = $(shell yosys-config --datdir/ice40/cells_sim.v)

help:
	@echo ""
	@echo "make sim       ... run simulation"
	@echo "make syn       ... run synthesis"
	@echo "make post      ... run post-synthesis simulation"
	@echo "make prog      ... program FPGA board"
	@echo "make ino       ... run Arduino-based test bench"
	@echo "make view      ... display simulation results"
	@echo ""
	@echo "prerequisites:"
	@echo "   Icarus Verilog, GtkWave, Python 3, Yosys, IceStorm, Arachne-pnr"
	@echo "   http://svn.clifford.at/tools/trunk/   (arduino-cc and teletype)"
	@echo ""

sim: chip.v
	iverilog -o chip_tb.exe -s chip_tb $(ice40sim) chip_tb.v chip.v spi_ctrl_*.v ico_*.v
	vvp -N chip_tb.exe +vcd

syn: chip.v
	yosys -p 'synth_ice40 -top chip -blif chip.blif' chip.v spi_ctrl_*.v ico_*.v
	arachne-pnr -d 8k -o chip.txt -p chip.pcf chip.blif

post:
	icebox_vlog -scp chip.pcf chip.txt > post.v
	iverilog -o post_tb.exe -s chip_tb $(ice40sim) chip_tb.v post.v
	vvp -N post_tb.exe +vcd

prog:
	icepack chip.txt chip.bin
	sudo iceprog -S chip.bin 

ino:
	arduino-cc -l SPI -P /dev/ttyACM0 spi_test.ino
	teletype /dev/ttyACM0 115200

spi_test: spi_test.cc
	gcc -o spi_test -Wall -Os spi_test.cc -lwiringPi

gpioprog: gpioprog.cc
	gcc -o gpioprog -Wall -Os gpioprog.cc -lwiringPi

test: spi_test
	gpio readall
	sudo ./spi_test

view:
	gtkwave chip_tb.vcd chip_tb.gtkw

chip.v: icogenerate.py config.txt
	python3 icogenerate.py

.PHONY: sim syn prog ino test view

