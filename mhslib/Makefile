
CC = gcc
CXX = gcc
CXXFLAGS = -ggdb -O3 -march=native -Wall -Wextra -MD -std=gnu++0x
LDLIBS = -lstdc++

# Profile using gprof
# Usage: ./puzzle3d; gprof ./puzzle3d
#CXXFLAGS += -pg
#LDFLAGS += -pg

# Profile using gcov
# Usage: ./puzzle3d; gcov mhslib.cc; cat mhslib.cc.gcov
#CXXFLAGS += -fprofile-arcs -ftest-coverage
#LDFLAGS += -fprofile-arcs -ftest-coverage

all: sudoku puzzle3d

sudoku: sudoku.o mhslib.o
puzzle3d: puzzle3d.o mhslib.o

clean:
	rm -f sudoku puzzle3d *.o *.d

.PHONY: all clean

-include *.d

