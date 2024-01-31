
CC = gcc
CFLAGS = -Wall -Os -ggdb -MD -pthread -std=c99
LDLIBS = -lftdi -lm

all: icestickprog icestickflash

icestickprog: icestickprog.c
	$(CC) -o $@ $(CFLAGS) $< $(LDLIBS) 

icestickflash: icestickflash.c
	$(CC) -o $@ $(CFLAGS) $< $(LDLIBS) 

install: icestickprog icestickflash
	install icestickprog /usr/local/bin/icestickprog
	install icestickflash /usr/local/bin/icestickflash
	install icecuberun.sh /usr/local/bin/icecuberun

clean:
	rm -f icestickprog icestickflash *.d

