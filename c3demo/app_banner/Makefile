
ifeq ($(shell which icoprog),)
SSH_RASPI := ssh pi@raspi
else
SSH_RASPI := sh -c
endif

all: appimage.hex

8x8font.h: 8x8font.py 8x8font.png
	python3 8x8font.py > 8x8font.h

appimage.elf: main.c 8x8font.h
	riscv32-unknown-elf-gcc --std=gnu99 -Os -m32 -o appimage.elf main.c
	chmod -x appimage.elf

appimage.hex: appimage.elf
	riscv32-unknown-elf-objcopy -O verilog appimage.elf appimage.hex
	chmod -x appimage.hex

run: appimage.hex
	$(SSH_RASPI) 'icoprog -b'
	$(SSH_RASPI) 'icoprog -zc2' < appimage.hex

.PHONY: all run

