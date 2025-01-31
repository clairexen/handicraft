CXX = clang++
CXXFLAGS = -O2 -ggdb -std=c++26 -Wall -MD
CC = $(CXX)
CCFLAGS = $(CXXFLAGS)

wdroid: wdroid.o words.o

run: wdroid
	./wdroid -4 -5 -6 -exit
	./wdroid -M float/00112 magot/BOAST -l -exit
	./wdroid --5 --K --R --M float/00112 magot/BOAST --l --exit

words.cc: words.py
	python3 words.py

clean:
	rm -f wdroid words.cc *.o *.d

-include *.d
