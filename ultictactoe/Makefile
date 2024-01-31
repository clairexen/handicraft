
CXX = clang
CXXFLAGS = -Wall -Wextra -Werror
# CXXFLAGS += -O0 -ggdb
CXXFLAGS += -O3 -DNDEBUG
LDLIBS = -lm

MXECXX = /usr/local/src/mxe/usr/bin/i686-pc-mingw32-g++

ultictactoe: ultictactoe.o

ultictactoe.exe: ultictactoe.cc
	$(MXECXX) -o ultictactoe.exe ultictactoe.cc

clean:
	rm -f ultictactoe ultictactoe.exe ultictactoe.o

