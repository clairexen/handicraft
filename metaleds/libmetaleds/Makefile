
CLFAGS = -Wall -O2 -march=pentium-mmx -ggdb
LDLIBS = -lSDL -lSDL_net
LDFLAGS = -ggdb

all: demo001 demo002 demo003 demo004 libmetaleds.o

demo001: demo001.o libmetaleds.o
demo002: demo002.o libmetaleds.o
demo003: demo003.o libmetaleds.o
demo004: demo004.o libmetaleds.o

clean:
	rm -f *.o demo001 demo002 demo003 demo004
	rm -f core core.*

.PHONY: all clean

