CXX = clang++
CXXFLAGS = -Os -Wall -MD
CC = $(CXX)
CCFLAGS = $(CXXFLAGS)

wdroid: wdroid.o words.o

words.cc: words.py
	python3 words.py

clean:
	rm -f wdroid words.cc *.o *.d

-include *.d
