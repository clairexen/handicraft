CC = clang
CFLAGS = -Wall -Wextra -Os -std=c99 -ggdb
LDLIBS = -lm

test: multiphaseavg_tb
	ulimit -c unlimited; ./multiphaseavg_tb < testbench.dat

multiphaseavg_tb: multiphaseavg_tb.o multiphaseavg.o

clean:
	rm -f multiphaseavg_tb *.o *.d core

-include *.d

.PHONY: test clean
