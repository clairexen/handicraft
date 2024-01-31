#!/bin/bash
set -ex
build() {
	avr-gcc -Wall -Wextra -Os -mmcu=atmega328p -DF_CPU=16000000L -std=gnu99 -x c -o $1.elf $1.c
	avr-objcopy -j .text -j .data -O ihex $1.elf $1.hex
	avr-size $1.elf $1.hex
}
build histogram
build receiver
build sender
