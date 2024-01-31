
CXX = g++
CXXFLAGS += -Wall -I/usr/include/eigen2 -ggdb -O0 -MD
LDLIBS += -lm -lppm -lstdc++

cliprect: cliprect.o

clean:
	rm -f cliprect *.o *.d
	rm -f image3.ppm image4.ppm

-include *.d

