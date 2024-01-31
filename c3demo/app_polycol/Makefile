
ifeq ($(shell which icoprog),)
SSH_RASPI := ssh pi@raspi
else
SSH_RASPI := sh -c
endif

all: appimage.hex

appimage.elf: main.c syscalls.c irqwrapper.S
	riscv32-unknown-elf-gcc --std=gnu99 -O3 -m32 -march=RV32IXcustom -o appimage.elf main.c syscalls.c irqwrapper.S
	chmod -x appimage.elf

appimage.hex: appimage.elf
	riscv32-unknown-elf-objcopy -O verilog appimage.elf appimage.hex
	chmod -x appimage.hex

prog_flash: appimage.hex
	{ cat appimage.hex; echo '*'; } | $(SSH_RASPI) 'icoprog -f -O 4'

run: appimage.hex
	$(SSH_RASPI) 'icoprog -b'
	$(SSH_RASPI) 'icoprog -zc2' < appimage.hex

.PHONY: all run

