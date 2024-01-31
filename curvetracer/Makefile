
ARDUINO_TTY = /dev/ttyUSB0

CFLAGS = -Wall -Wextra -Werror -ggdb -Os -MD

all: firmware.hex calib show diode npn opamp

firmware.hex: firmware.c
	rm -f firmware.elf firmware.hex
	avr-gcc -MD -mmcu=atmega328p -DF_CPU=16000000L -Wall -Wextra -Os -o firmware.elf firmware.c
	avr-objcopy -j .text -j .data -O ihex firmware.elf firmware.hex
	avr-size firmware.elf firmware.hex

prog: firmware.hex
	avrdude -p m328p -c arduino -P $(ARDUINO_TTY) -b57600 -v -U flash:w:firmware.hex

calib: calib.o libcurvetr.o
show: show.o libcurvetr.o
diode: diode.o libcurvetr.o
npn: npn.o libcurvetr.o
opamp: opamp.o libcurvetr.o

clean:
	rm -vf calib show diode npn *.o *.d
	rm -vf firmware.elf firmware.hex

-include *.d

