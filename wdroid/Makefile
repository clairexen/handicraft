CXX = clang++
CXXFLAGS = -O2 -ggdb -std=c++26 -Wall -MD
CC = $(CXX)
CCFLAGS = $(CXXFLAGS)

wdroid: wdroid.o minmax.o words.o

run: wdroid
	./wdroid -4 -5 -6 -exit
	./wdroid -M float/00112 magot/BOAST -l -i --K --exit

run-minmax-aloes: wdroid
	time ./wdroid -minmax +first=aloes +go -exit

run-minmax-store: wdroid
	time ./wdroid -minmax +first=store +go -exit

run-hella: wdroid
	./wdroid aloes/PEDAL learn/PEDAL -l -minmax-q +go \
		-reset +word=hella -minmax-q +first=hella +go -exit

run-ptdemo: ptdata.txt .venv/stamp
	.venv/bin/python3 ptdemo.py

words.cc: words.py wordledb.json
	python3 words.py

wordledb.json: wordledb.py
	python3 wordledb.py

ptdata.txt: wdroid
	./wdroid -3 -minmax +go +wrDatFile=ptdata.txt -exit

.venv/stamp:
	python3 -m venv .venv
	.venv/bin/pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	touch .venv/stamp

clean:
	rm -f wdroid words.cc *.o *.d

-include *.d
