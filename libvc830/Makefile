
CXX = g++
CXXFLAGS = -Os -MD
LDLIBS = -lhidapi-libusb -lstdc++ -lm

vc830demo: libvc830.o vc830demo.o

clean:
	rm -f vc830demo *.o *.d

-include *.d

