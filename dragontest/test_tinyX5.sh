#!/bin/bash

set -ex

# AVRTYPE="25"
# AVRTYPE="45"
AVRTYPE="85"

AVR_CC="avr-gcc"
AVR_OBJCOPY="avr-objcopy"
AVR_CFLAGS="-Wall -Werror -Os -std=gnu99 -mmcu=attiny$AVRTYPE"

EFUSE_BITS="ff" # bxxxxxxx1
HFUSE_BITS="df" # b11011111
LFUSE_BITS="62" # b01100010
LOCK_BITS="ff"  # bxxxxxx11

$AVR_CC $AVR_CFLAGS test_blink.c -o test_tiny$AVRTYPE.elf
$AVR_OBJCOPY -O ihex -R .eeprom test_tiny$AVRTYPE.elf test_tiny$AVRTYPE.hex

sudo avrdude -c dragon_hvsp -p t$AVRTYPE -P usb -U flash:w:test_tiny$AVRTYPE.hex \
		-U efuse:w:0x$EFUSE_BITS:m -U hfuse:w:0x$HFUSE_BITS:m \
		-U lfuse:w:0x$LFUSE_BITS:m -U lock:w:0x$LOCK_BITS:m

