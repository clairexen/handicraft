# The New York Times "WordleBot" is behnd a paywall.  :/
# So I wrote my own "WordleDroid" which I can run locally.
#
# Copyright (C) 2025  Claire Xenia Wolf <claire@clairexen.net>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


#####################################################
# Build Configuration

MKANN = 0
MKWORDS = 0
MK5ONLY = 0
MKSTATIC = 0

CXX = clang++
CXXFLAGS = -O2 -ggdb -std=c++26 -Wall -MD
LD = $(CXX)
LDFLAGS =
LDLIBS =

ifeq ($(MK5ONLY),1)
CXXFLAGS += -DENABLE_WDROID_ENGINE_5
else
CXXFLAGS += -DENABLE_WDROID_ENGINE_3
CXXFLAGS += -DENABLE_WDROID_ENGINE_4
CXXFLAGS += -DENABLE_WDROID_ENGINE_5
CXXFLAGS += -DENABLE_WDROID_ENGINE_6
endif

ifeq ($(MKSTATIC),1)
LDFLAGS += -static
endif

CXXFLAGS += $(shell pkg-config --cflags libedit)
LDLIBS += $(shell pkg-config --libs libedit)


#####################################################
# WordleDroid C++ Build

CXXFLAGS_O3 = -O3 -DNDEBUG $(filter-out -O2,$(CXXFLAGS))

OBJS = wdroid.o minmax.o words.o anneval.o

wdroid: $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

wdroid_o3: $(OBJS:.o=_o3.o)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%_o3.o: %.cc
	$(CXX) $(CXXFLAGS_O3) -c -o $@ $<


#####################################################
# Run Various Examples

run: wdroid
	./wdroid -3 -4 -5 -6 -exit
	./wdroid -5 -M -i float/00112 magot/01012 -l -i ."-K -exit"

run-minmax-aloes: wdroid
	time ./wdroid -minmax +first=aloes +go -exit

run-minmax-store: wdroid
	time ./wdroid -minmax +first=store +go -exit

run-hella: wdroid
	./wdroid aloes/learn/12100 -l -minmax-q +go \
		-reset +word=hella -minmax-q +first=hella +go -exit


#####################################################
# Word Lists

ifeq ($(MKWORDS),1)
words.cc: words.py wordledb.json .venv/stamp-nltk
	# apt install wamerican-english-large
	.venv/bin/python3 words.py

wordledb.json: wordledb.py .venv/stamp-requests
	.venv/bin/python3 wordledb.py
else
words.cc:
	$(MAKE) MKWORDS=1 words.cc
endif


#####################################################
# ANN Training for 3-letter words (with PyTorch)

ifeq ($(MKANN),1)
ptmodel3.bin: wdroid .venv/stamp-torch
	time ./wdroid -3 -minmax +go +wrDatFile=ptdata3A0.txt \
		--system=".venv/bin/python3 ptdriver.py -o ptmodel3A -e 8 -H 50 ptdata3A0.txt" \
		+useAnnModel=ptmodel3A.bin +maxDatMBs=1 +wrDatFile=ptdata3B0.txt \
		--system=".venv/bin/python3 ptdriver.py -o ptmodel3B -i ptmodel3A -e 8 ptdata3B0.txt" \
		+useAnnModel=ptmodel3B.bin +maxDatMBs=1 +wrDatFile=ptdata3C0.txt \
		--system=".venv/bin/python3 ptdriver.py -o ptmodel3C -i ptmodel3B -e 8 ptdata3C0.txt" \
		+useAnnModel=ptmodel3C.bin +maxDatMBs=1 +wrDatFile=ptdata3D0.txt \
		-quit
	time .venv/bin/python3 ptdriver.py -o ptmodel3 -i ptmodel3C \
		ptdata3[A-C]0.txt ptdata3[A-C]0.txt ptdata3A0.txt
else
ptmodel3.bin:
	$(MAKE) MKANN=1 ptmodel3.bin
endif


#####################################################
# ANN Training for 5-letter words (with PyTorch)

ptmodel5A.bin: wdroid .venv/stamp-torch
	rm -f ptdata5A[0-9].txt
	time ./wdroid -5 -minmax +frnag +f=carol +f=final +f=topee +f=abort +f=leapt +f=sepal \
		+f=hoard +f=smart +f=fecal +f=depot +f=scorn +f=spoil +f=aleph +f=shard +setup +go \
		+maxDatMBs=500 +maxDatFiles=10 +wrDatFile=ptdata5A%.txt -quit
	time .venv/bin/python3 ptdriver.py -o ptmodel5A ptdata5A[0-9].txt

ptmodel5B.bin: ptmodel5A.bin wdroid .venv/stamp-torch
	rm -f ptdata5B[0-9].txt
	time ./wdroid -5 -minmax +f=scald +f=impel +f=asdic +f=faced +f=dumpy +setup +go \
		+useAnnModel=ptmodel5A.bin +maxDatMBs=500 +maxDatFiles=10 +wrDatFile=ptdata5B%.txt -quit
	time .venv/bin/python3 ptdriver.py -o ptmodel5B -i ptmodel5A ptdata5B[0-9].txt

ptmodel5C.bin: ptmodel5B.bin wdroid .venv/stamp-torch
	rm -f ptdata5C[0-9].txt
	time ./wdroid -5 -seed=1 -minmax +frnag +fgrs=10 +frrs=10 +setup +go \
		+useAnnModel=ptmodel5B.bin +maxDatMBs=500 +maxDatFiles=10 +wrDatFile=ptdata5C%.txt -quit
	time .venv/bin/python3 ptdriver.py -o ptmodel5C -i ptmodel5B ptdata5C[0-9].txt

ptmodel5D.bin: ptmodel5C.bin wdroid .venv/stamp-torch
	rm -f ptdata5D[0-9].txt
	time ./wdroid -5 -seed=2 -minmax +frag +fgrs=10 +setup +go \
		+useAnnModel=ptmodel5C.bin +maxDatMBs=500 +maxDatFiles=10 +wrDatFile=ptdata5D%.txt -quit
	time .venv/bin/python3 ptdriver.py -o ptmodel5D -i ptmodel5C ptdata5D[0-9].txt

ifeq ($(MKANN),1)
ptmodel5.bin: ptmodel5D.bin .venv/stamp-torch
	time .venv/bin/python3 ptdriver.py -o ptmodel5 -i ptmodel5D -e 2 ptdata5[A-D][0-9].txt
else
ptmodel5.bin:
	$(MAKE) MKANN=1 ptmodel5.bin
endif


#####################################################
# Python Virtual-Env

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

.venv/stamp-ipython: .venv/stamp
	.venv/bin/pip3 install ipython
	touch .venv/stamp-ipython

ipython: .venv/stamp-ipython
	.venv/bin/ipython3


#####################################################
# Install & Clean

install: wdroid
	install -st /usr/local/bin wdroid

clean:
	rm -f wdroid_o3 *.o *.d
	rm -f ptdata[3456][A-Z][0-9].txt
	rm -f ptmodel[3456][A-Z].*

purge: clean
	rm -f wdroid
	rm -f words.cc
	rm -f ptmodel[3456].*

mrproper: purge
	rm -rf .venv/

-include *.d
