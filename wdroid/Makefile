CXX = clang++
CXXFLAGS = -O2 -ggdb -std=c++26 -Wall -MD
CXXFLAGS_O3 = -O3 -DNDEBUG $(filter-out -O2,$(CXXFLAGS))

OBJS = wdroid.o minmax.o words.o anneval.o

wdroid: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

wdroid_o3: $(OBJS:.o=_o3.o)
	$(CXX) $(CXXFLAGS_O3) -o $@ $^

%_o3.o: %.cc
	$(CXX) $(CXXFLAGS_O3) -c -o $@ $<

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

words.cc: words.py wordledb.json .venv/stamp-nltk
	# apt install wamerican-english-large
	.venv/bin/python3 words.py

wordledb.json: wordledb.py .venv/stamp-requests
	.venv/bin/python3 wordledb.py

ptmodel0.bin: ptdata0.txt .venv/stamp-torch
	.venv/bin/python3 ptdriver.py -o ptmodel0 ptdata[0-7].txt

ptdata0.txt: wdroid
	time ./wdroid -5 -minmax +limit=350 +go \
		+maxDatMBs=500 +maxDatFiles=8 +wrDatFile=ptdata%.txt -exit

ptdata8.txt: wdroid ptmodel0.bin
	time ./wdroid -5 -minmax +limit=350 +go \
		+maxDatMBs=500 +useAnnModel=ptmodel0.bin +wrDatFile=ptdata8.txt -exit

ptmodel8.bin: ptdata8.txt .venv/stamp-torch
	.venv/bin/python3 ptdriver.py -i ptmodel0.bin -o ptmodel8 ptdata8.txt

ptdata9.txt: wdroid ptmodel8.bin
	time ./wdroid -5 -minmax +limit=350 +go \
		+maxDatMBs=500 +useAnnModel=ptmodel8.bin +wrDatFile=ptdata9.txt -exit

ptmodel9.bin: ptdata9.txt .venv/stamp-torch
	.venv/bin/python3 ptdriver.py -i ptmodel8.bin -o ptmodel9 ptdata9.txt

.venv/stamp:
	python3 -m venv .venv
	touch .venv/stamp

.venv/stamp-nltk: .venv/stamp
	.venv/bin/pip3 install nltk
	touch .venv/stamp-nltk

.venv/stamp-requests: .venv/stamp
	.venv/bin/pip3 install requests
	touch .venv/stamp-requests

.venv/stamp-torch: .venv/stamp
	.venv/bin/pip3 install torch torchvision torchaudio
	touch .venv/stamp-torch

clean:
	rm -f wdroid wdroid_o3 *.o *.d
	rm -f ptdata[0-9].txt ptmodel[089].*
	rm -f words.cc

-include *.d
