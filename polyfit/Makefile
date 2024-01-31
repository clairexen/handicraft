
CXX = clang
CXXFLAGS  = -Wall -Os -std=c++11
CXXFLAGS += -Wno-deprecated-register
CXXFLAGS += -I/usr/local/include/eigen3
LDLIBS += -lm -lstdc++

polyfit: polyfit.o

clean:
	rm -f polyfit polyfit.o
