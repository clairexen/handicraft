
CC = gcc
CFLAGS += -Wall -ggdb -O0 -MD
LDLIBS += -lm

demo: demo.o nmosaoi.tab.o

demo.o: nmosaoi.tab.c
nmosaoi.tab.c: nmosaoi.tnl tnltool
	../../src/tnltool nmosaoi

tnltool:
	make -C ../../src

clean:
	rm -f nmosaoi.tab.c nmosaoi.tab.h
	rm -f *.d *.o demo tnltool

.PHONY: tnltool clean

-include *.d

