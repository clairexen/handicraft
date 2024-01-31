#!/bin/bash
set -xe
avr-gcc -Wall -std=gnu99 -O3 -o recv.elf -mmcu=atmega328p -DF_CPU=16000000L recv.c
avr-objcopy -j .text -j .data -O ihex recv.elf recv.hex
avrdude -p m328p -b 115200 -c arduino -P /dev/ttyACM0 -v -U flash:w:recv.hex
teletype /dev/ttyACM0 2000000
