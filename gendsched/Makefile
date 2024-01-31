
CXX = clang
CXXFLAGS = -std=c++11 -ggdb -MD
LDLIBS = -lstdc++

gendsched: parser.o

mktest: mktest.o

test: gendsched mktest
	./mktest 123 > example.txt
	./gendsched example.txt

clean:
	rm -f mktest *.o *.d

-include *.d

