
CXX = g++
CXXFLAGS += -std=gnu++0x -Wall -ggdb -O0 -MD

demo: CC=$(CXX)
demo: demo.o nmossim.o display.o model6502.o

clean:
	rm -f demo *.o *.d nmossim_disp* *~

-include *.d

