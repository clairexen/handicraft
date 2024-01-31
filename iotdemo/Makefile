
ifeq ($(shell which icoprog),)
SSH_RASPI := ssh pi@raspi
else
SSH_RASPI := sh -c
endif

help:
	@echo ""
	@echo "Building FPGA bitstream and program:"
	@echo "   make prog_sram"
	@echo "   make prog_flash"
	@echo ""
	@echo "Building firmware image and update:"
	@echo "   make prog_firmware"
	@echo ""
	@echo "Resetting FPGA (prevent boot from flash):"
	@echo "   make reset_halt"
	@echo ""
	@echo "Resetting FPGA (load image from flash):"
	@echo "   make reset_boot"
	@echo ""
	@echo "Build and upload FPGA + application image:"
	@echo "   make run"
	@echo ""
	@echo "Console session (close with Ctrl-D):"
	@echo "   make console"
	@echo ""
	@echo "Build and run testbench:"
	@echo "   make testbench"
	@echo ""
	@echo "Download debug trace (to 'debug.vcd'):"
	@echo "   make debug"
	@echo ""

firmware.elf: firmware.S firmware.c firmware.lds
	riscv32-unknown-elf-gcc -Os -m32 -ffreestanding -nostdlib -Wall -o firmware.elf firmware.S firmware.c \
			--std=gnu99 -Wl,-Bstatic,-T,firmware.lds,-Map,firmware.map,--strip-debug -lgcc
	chmod -x firmware.elf

firmware.bin: firmware.elf
	riscv32-unknown-elf-objcopy -O binary firmware.elf firmware.bin
	chmod -x firmware.bin

firmware.hex: makehex.py firmware.bin
	python3 makehex.py firmware.bin 1024 > firmware.hex
	@echo "Firmware size: $$(grep .. firmware.hex | wc -l) / $$(wc -l < firmware.hex)"

appimage.elf: demoapp.c
	riscv32-unknown-elf-gcc --std=gnu99 -Os -m32 -Wall -o appimage.elf demoapp.c
	chmod -x appimage.elf

appimage.hex: appimage.elf
	riscv32-unknown-elf-objcopy -O verilog appimage.elf appimage.hex
	chmod -x appimage.hex

iotdemo.blif: iotdemo.v pmods.v picorv32.v firmware.hex
	yosys -v2 -p 'synth_ice40 -abc2 -top iotdemo -blif iotdemo.blif' iotdemo.v pmods.v picorv32.v

iotdemo.asc: iotdemo.pcf iotdemo.blif
	set -x; for seed in 1234 2345 4567 5678 6789; do \
		arachne-pnr -s $$seed -d 8k -p iotdemo.pcf -o iotdemo.asc iotdemo.blif && exit 0; \
	done; false

iotdemo.bin: iotdemo.asc
	icepack iotdemo.asc iotdemo.bin

iotdemo.rpt: iotdemo.asc
	icetime -d hx8k -tr iotdemo.rpt iotdemo.asc

prog_sram: iotdemo.bin
	$(SSH_RASPI) 'icoprog -p' < iotdemo.bin

prog_flash: iotdemo.bin
	$(SSH_RASPI) 'icoprog -f' < iotdemo.bin

prog_firmware: firmware.bin
	$(SSH_RASPI) 'icoprog -w1' < firmware.bin

reset_halt:
	$(SSH_RASPI) 'icoprog -R'

reset_boot:
	$(SSH_RASPI) 'icoprog -b'

run: iotdemo.bin appimage.hex
	$(SSH_RASPI) 'icoprog -p' < iotdemo.bin
	$(SSH_RASPI) 'icoprog -zc2' < appimage.hex

console:
	$(SSH_RASPI) 'icoprog -c2'

testbench.exe: firmware.hex testbench.v iotdemo.v pmods.v picorv32.v
	iverilog -o testbench.exe -s testbench testbench.v iotdemo.v pmods.v picorv32.v \
	                $(shell yosys-config --datdir/ice40/cells_sim.v)
	chmod -x testbench.exe

testbench: testbench.exe appimage.hex
	vvp -N testbench.exe

debug:
	sedexpr="$$( grep '//.*debug_.*->' iotdemo.v | sed 's,.*\(debug_\),s/\1,; s, *-> *, /,; s, *$$, /;,;'; )"; \
	$(SSH_RASPI) 'icoprog -V31' | sed -e "$$sedexpr" > debug.vcd

clean:
	rm -f iotdemo.bin
	rm -f iotdemo.blif
	rm -f iotdemo.asc
	rm -f iotdemo.rpt
	rm -f firmware.bin
	rm -f firmware.elf
	rm -f firmware.hex
	rm -f firmware.map
	rm -f appimage.elf
	rm -f appimage.hex
	rm -f testbench.exe
	rm -f testbench.vcd

.SECONDARY:
.PHONY: prog_sram prog_flash prog_firmware reset_halt reset_boot run console debug clean

