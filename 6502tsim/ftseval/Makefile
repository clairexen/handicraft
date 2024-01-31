
CC = gcc
# CFLAGS += -Wall -O0 -ggdb -MD
CFLAGS += -Wall -O3 -DNDEBUG -MD

demo: demo.o ftseval.o model6502.o

clean:
	rm -f demo *.o *.d *~

-include *.d

