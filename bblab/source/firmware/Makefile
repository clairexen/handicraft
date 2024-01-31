
PLATFORM = bblab
# PLATFORM = arduino


ifeq ($(PLATFORM),bblab)
	DEVICE  = attiny85
	F_CPU   = 16500000 # in Hz
	MEMSIZE = 380

	############################## ATTiny25/45/85 ###############################
	# go to http://palmavr.sourceforge.net/cgi-bin/fc.cgi and choose your settings
	#
	# ATtiny*5 FUSE_L (Fuse low byte):
	# 0xe1 = 1 1 1 0   0 0 0 1
	#        ^ ^ \+/   \--+--/
	#        | |  |       +------- CKSEL 3..0 (clock selection -> HF PLL)
	#        | |  +--------------- SUT 1..0 (BOD enabled, fast rising power)
	#        | +------------------ CKOUT (clock output on CKOUT pin -> disabled)
	#        +-------------------- CKDIV8 (divide clock by 8 -> don't divide)
	#
	# ATtiny*5 FUSE_H (Fuse high byte):
	# 0xfd = 0 1 1 1   1 1 0 1
	#        ^ ^ ^ ^   ^ \-+-/ 
	#        | | | |   |   +------ BODLEVEL 2..0 (brownout trigger level -> 2.7V)
	#        | | | |   +---------- EESAVE (preserve EEPROM on Chip Erase -> not preserved)
	#        | | | +-------------- WDTON (watchdog timer always on -> disable)
	#        | | +---------------- SPIEN (enable serial programming -> disabled)
	#        | +------------------ DWEN (debug wire enable)
	#        +-------------------- RSTDISBL (disable external reset -> enabled)
	#
	FUSE_L = e1 # set to 0xa1 to monitor clock
	FUSE_H = 7d

	AVRBOOTPROG_TTY = /dev/ttyUSB0
	PRGCMD = avrbootprog /dev/ttyUSB0 $(DEVICE) chipdel:w: flash:w:firmware.hex lfuse:w:$(FUSE_L) hfuse:w:$(FUSE_H) efuse:w:ff
	CLIENTCMD = @echo 'Have a nice day!'
endif

ifeq ($(PLATFORM),arduino)
	DEVICE  = atmega168
	F_CPU   = 16000000 # in Hz
	MEMSIZE = 256

	ARDUINO_TTY = /dev/ttyUSB0
	PRGCMD = avrdude -p m328p -c arduino -P $(ARDUINO_TTY) -b57600 -v -U flash:w:firmware.hex
	CLIENTCMD = ! teletype $(ARDUINO_TTY) 19200
endif


CC = avr-gcc
CFLAGS = -std=gnu99 -Wall -O3 -DF_CPU=$(F_CPU) -DPLATFORM_$(PLATFORM)=1 -DMEMSIZE=$(MEMSIZE) -mmcu=$(DEVICE)
LDFLAGS =
OBJS = firmware.o

# usbdrv
CFLAGS += -Iusbdrv -I. -DDEBUG_LEVEL=0
OBJS += usbdrv/usbdrv.o usbdrv/usbdrvasm.o usbdrv/oddebug.o


help:
	@echo ""
	@echo "Usage:"
	@echo "  make firmware.hex     Just build it"
	@echo "  make prog             Build from scratch and program"
	@echo "  make clean            Remove all generated files"
	@echo ""


.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

.S.o:
	$(CC) $(CFLAGS) -x assembler-with-cpp -c $< -o $@

firmware.elf: $(OBJS)
	$(CC) $(LDFLAGS) -o firmware.elf $(OBJS)

firmware.hex: firmware.elf
	rm -f firmware.hex
	avr-objcopy -j .text -j .data -O ihex firmware.elf firmware.hex
	avr-size firmware.elf firmware.hex

prog:
	make clean && make firmware.hex
	$(PRGCMD)
	$(CLIENTCMD)

clean:
	rm -f firmware.hex firmware.elf firmware.o *.o usbdrv/*.o

