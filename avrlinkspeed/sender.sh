#!/bin/bash
set -xe
avr-gcc -Wall -std=gnu99 -O3 -o sender.elf -mmcu=atmega328p -DF_CPU=16000000L sender.c
avr-objcopy -j .text -j .data -O ihex sender.elf sender.hex
avrdude -p m328p -b 115200 -c arduino -P /dev/ttyACM0 -v -U flash:w:sender.hex
