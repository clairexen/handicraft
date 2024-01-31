
CXX = g++
CXXFLAGS += -MD -Wall -Wextra -Os -ggdb -std=gnu++0x
LDLIBS += -lstdc++

all: vlnlp-demo

vlnlp-demo: vlnlp-demo.o libvlnlp.a

libvlnlp.a: vlnlp.o parser.tab.o lexer.o
	rm -f $@
	ar qc $@ $^
	ranlib $@

parser.tab.cc parser.tab.h: parser.y
	bison -d -r all parser.y
	mv parser.tab.c parser.tab.cc

lexer.cc: lexer.l
	flex -o lexer.cc lexer.l

clean:
	rm -f vlnlp-demo *.d *.o core
	rm -f parser.tab.cc parser.tab.h parser.output lexer.cc

-include *.d

