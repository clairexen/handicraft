
VERIFIC_DIR = /usr/local/src/verific_lib_eval
VERIFIC_COMPONENTS = verilog vhdl database util containers sdf

CC = gcc
CXX = gcc
CXXFLAGS = -O0 -m32 -ggdb $(patsubst %,-I$(VERIFIC_DIR)/%,$(VERIFIC_COMPONENTS))
CXXFLAGS += -D'VERIFIC_DIR="$(VERIFIC_DIR)"'
LDLIBS = $(patsubst %,$(VERIFIC_DIR)/%/*-linux.a,$(VERIFIC_COMPONENTS)) -lstdc++ -lm
LDFLAGS = -m32 -static

verifitest: verifitest.o

install: verifitest
	install verifitest /usr/local/bin/verifitest

test: ./verifitest
	./verifitest -i test.txt -o test.v test.vhd
	yosys -p 'read_verilog -lib cells.v; hierarchy -top test; show test' test.v

clean:
	rm -f verifitest verifitest.o test.v test.txt

