
CXX = g++
CXXFLAGS += -std=gnu++0x -Wall -ggdb -O0 -MD
LDLIBS += -lm -lstdc++

tnltool: tnltool.o tnl_parser.tab.o tnl_lexer.tab.o \
		checks.o graph.o bias.o backend_csim.o

tnl_parser.tab.cc tnl_parser.tab.h: tnl_parser.y
	bison -d tnl_parser.y
	mv tnl_parser.tab.c tnl_parser.tab.cc

tnl_lexer.tab.cc: tnl_lexer.l
	flex -o tnl_lexer.tab.cc tnl_lexer.l

clean:
	rm -f tnl_parser.tab.* tnl_lexer.tab.*
	rm -f *.o *.d tnltool

-include *.d

