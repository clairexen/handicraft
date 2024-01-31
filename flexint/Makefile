CXX = clang
CC = $(CXX)
LDLIBS = -lm -lstdc++
CXXFLAGS = -Ibigint/ -O0 -ggdb -MD -std=c++11

hello: hello.o bigint/BigInteger.o bigint/BigUnsigned.o

clean:
	rm -f bigint/*.o bigint/*.d
	rm -f hello *.o *.d

-include bigint/*.d
-include *.d

