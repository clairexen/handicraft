
PREFIX = /usr/local
BINDIR = $(PREFIX)/bin

CXX = clang
CXXFLAGS += -MD -Wall -ggdb -std=c++11
LDFLAGS += -ggdb
LDLIBS += -lstdc++ -lm -lminisat

# CXXFLAGS += -I/usr/include/pgapack-serial/ -DWL=64 -DFAKE_MPI
# LDLIBS += -lpgapack-serial1

dfslang: dfslang.o parser.o lexer.o ast.o prog.o sched.o map.o \
		sched_greedy.o sched_minisat.o map_vlog.o

test: dfslang
	cd examples/det3 && bash run.sh
	cd examples/sincpde && bash run_cores.sh
	cd examples/sincpde && bash run_multi.sh

parser.cc: parser.y
	bison -dv -o parser.cc parser.y

lexer.cc: lexer.l
	flex -o lexer.cc lexer.l

clean:
	rm -f parser.cc parser.hh parser.output
	rm -f *.o *.d lexer.cc dfslang

install: dfslang
	install dfslang $(BINDIR)/dfslang

uninstall:
	rm -f $(BINDIR)/dfslang

-include *.d

