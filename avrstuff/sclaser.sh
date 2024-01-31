#!/bin/bash
set -xe
avr-gcc -Wall -std=gnu99 -O3 -o sclaser.elf -mmcu=atmega328p -DF_CPU=16000000L sclaser.c
avr-objcopy -j .text -j .data -O ihex sclaser.elf sclaser.hex
sudo avrdude -c dragon_pp -p m168 -P usb -v -U flash:w:sclaser.hex
