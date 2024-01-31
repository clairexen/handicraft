
CC = gcc
CFLAGS += -Wall -ggdb -O0 -MD
LDLIBS += -lm

demo: demo.o model6502.tab.o

demo.o: model6502.tab.c
model6502.tab.c: model6502.tnl model6502.inc tnltool
	../../src/tnltool model6502

tnltool:
	make -C ../../src

clean:
	rm -f model6502.tab.c model6502.tab.h
	rm -f *.d *.o demo

.PHONY: tnltool clean

-include *.d

