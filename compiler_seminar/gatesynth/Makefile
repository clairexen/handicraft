
CXX = clang
CXXFLAGS = -std=c++11 -MD -ggdb
LDLIBS += -lstdc++

gatesynth: gatesynth.o parser.tab.o lexer.o aitree.o

parser.tab.cc parser.tab.h: parser.y
	bison -d -r all -b parser parser.y
	mv parser.tab.c parser.tab.cc

lexer.cc: lexer.l
	flex -o lexer.cc lexer.l

test: gatesynth
	./gatesynth "xor1 = (a+b)*(a*b)'; xor2 = (a*b')+(a'*b);"

clean:
	rm -f gatesynth *.o *.d parser.tab.cc parser.tab.h parser.output lexer.cc

-include *.d

