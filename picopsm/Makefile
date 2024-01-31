
CC = clang
CFLAGS = -Wall -O0 -ggdb -MD -std=c11

test: picopsm
	./picopsm -asm example_001.asm > example_001.v
	iverilog -o testbench testbench.v picopsm.v
	./testbench

picopsm: picopsm.o picopsm_asm.o

clean:
	rm -vf picopsm testbench *.o *.d

-include *.d

