
ifeq ($(shell which icoprog),)
SSH_RASPI := ssh pi@raspi
else
SSH_RASPI := sh -c
endif

run: icobrdtst.bin
	$(SSH_RASPI) 'icoprog -p' < icobrdtst.bin
	$(SSH_RASPI) 'icoprog -c2'

firmware: firmware.bin
	$(SSH_RASPI) 'icoprog -w1' < firmware.bin
	$(SSH_RASPI) 'icoprog -c2'

firmware.elf: firmware.S firmware.c firmware.lds
	riscv32-unknown-elf-gcc -Os -m32 -ffreestanding -nostdlib -o firmware.elf firmware.S firmware.c \
			--std=gnu99 -Wl,-Bstatic,-T,firmware.lds,-Map,firmware.map,--strip-debug -lgcc
	chmod -x firmware.elf

firmware.bin: firmware.elf
	riscv32-unknown-elf-objcopy -O binary firmware.elf firmware.bin
	chmod -x firmware.bin

firmware.hex: makehex.py firmware.bin
	python3 makehex.py firmware.bin 1024 > firmware.hex
	@echo "Firmware size: $$(grep .. firmware.hex | wc -l) / $$(wc -l < firmware.hex)"

icobrdtst.blif: icobrdtst.v picorv32.v firmware.hex
	yosys -v2 -p 'synth_ice40 -abc2 -top icobrdtst -blif icobrdtst.blif' icobrdtst.v picorv32.v

icobrdtst.asc: icobrdtst.pcf icobrdtst.blif
	arachne-pnr -s 2 -d 8k -p icobrdtst.pcf -o icobrdtst.asc icobrdtst.blif

icobrdtst.bin: icobrdtst.asc
	icepack icobrdtst.asc icobrdtst.bin

icobrdtst.rpt: icobrdtst.asc
	icetime -d hx8k -tr icobrdtst.rpt icobrdtst.asc

clean:
	rm -f icobrdtst.bin
	rm -f icobrdtst.blif
	rm -f icobrdtst.asc
	rm -f icobrdtst.rpt
	rm -f firmware.bin
	rm -f firmware.elf
	rm -f firmware.hex
	rm -f firmware.map

.SECONDARY:
.PHONY: run firmware clean

