
ARDUINO_TTY = /dev/ttyUSB0

ARDUINO_DIR = /usr/share/arduino
ARDUINO_CORELIB_DIR = $(ARDUINO_DIR)/hardware/arduino/cores/arduino

CXXFLAGS = -MD -g -w -fno-exceptions -ffunction-sections -fdata-sections
CXXFLAGS += -mmcu=atmega328p -DF_CPU=16000000L -DARDUINO=18
CXXFLAGS += -I$(ARDUINO_CORELIB_DIR)

LDFLAGS = -Wl,--gc-sections -mmcu=atmega328p -L.
LDLIBS = -lArduinoMegaCore -lm

ARDUINO_CORELIB_OBJS += $(addprefix libArduinoMegaCore_,$(addsuffix .o,$(basename $(notdir $(wildcard $(ARDUINO_CORELIB_DIR)/*.c)))))
ARDUINO_CORELIB_OBJS += $(addprefix libArduinoMegaCore_,$(addsuffix .o,$(basename $(notdir $(wildcard $(ARDUINO_CORELIB_DIR)/*.cpp)))))

OBJS = firmware.o

firmware.hex: firmware.elf
	rm -f firmware.hex
	avr-objcopy -j .text -j .data -O ihex firmware.elf firmware.hex
	avr-size firmware.elf firmware.hex

firmware.elf: $(OBJS) libArduinoMegaCore.a
	avr-gcc $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

%.o: %.cc
	avr-g++ -c $(CXXFLAGS) -o $@ $<

libArduinoMegaCore.a: $(ARDUINO_CORELIB_OBJS)
	rm -f $@
	avr-ar -q $@ $^
	avr-ranlib $@

libArduinoMegaCore_%.o: $(ARDUINO_CORELIB_DIR)/%.c
	avr-gcc -c $(CXXFLAGS) -o $@ $<

libArduinoMegaCore_%.o: $(ARDUINO_CORELIB_DIR)/%.cpp
	avr-g++ -c $(CXXFLAGS) -o $@ $<

prog: firmware.hex
	avrdude -p m328p -c arduino -P $(ARDUINO_TTY) -b57600 -v -U flash:w:firmware.hex

clean:
	rm -vf libArduinoMegaCore.a *.o *.d
	rm -vf firmware.elf firmware.hex

-include *.d

