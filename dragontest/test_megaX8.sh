#!/bin/bash

set -ex

# AVRTYPE="48"
# AVRTYPE="88"
AVRTYPE="168"
# AVRTYPE="328"

AVR_CC="avr-gcc"
AVR_OBJCOPY="avr-objcopy"
AVR_CFLAGS="-Wall -Werror -Os -std=gnu99 -mmcu=atmega$AVRTYPE"

EFUSE_BITS="f9" # bxxxxx001
HFUSE_BITS="df" # b11011111
LFUSE_BITS="62" # b01100010
LOCK_BITS="ff"  # bxx111111

$AVR_CC $AVR_CFLAGS test_blink.c -o test_mega$AVRTYPE.elf
$AVR_OBJCOPY -O ihex -R .eeprom test_mega$AVRTYPE.elf test_mega$AVRTYPE.hex

sudo avrdude -c dragon_pp -p m$AVRTYPE -P usb -U flash:w:test_mega$AVRTYPE.hex \
		-U efuse:w:0x$EFUSE_BITS:m -U hfuse:w:0x$HFUSE_BITS:m \
		-U lfuse:w:0x$LFUSE_BITS:m -U lock:w:0x$LOCK_BITS:m

