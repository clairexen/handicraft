
CC = clang
CXX = clang

CXXFLAGS = -std=c++11 -ggdb -O0 -MD
LDLIBS = -lstdc++ -lsystemc -lm

all: demo001

demo001: demo001.o

clean:
	rm -f demo001 *.o *.d *.vcd

-include *.d

