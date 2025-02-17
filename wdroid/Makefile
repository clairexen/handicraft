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
MKSTATIC = 0

CXX = clang++
CXXFLAGS = -O2 -ggdb -std=c++26 -Wall -MD
LD = $(CXX)
LDFLAGS =
LDLIBS =

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
# ANN Training (with PyTorch)

ifeq ($(MKANN),1)
ptmodel5.bin: ptmodel5C.bin .venv/stamp-torch
	.venv/bin/python3 ptdriver.py -o ptmodel5 -i ptmodel5C -e 2 ptdata5[A-Z][0-9].txt

ptmodel5C.bin: wdroid .venv/stamp-torch
	time ./wdroid -5 -minmax +limit=350 +go \
		+maxDatMBs=500 +maxDatFiles=4 +wrDatFile=ptdata5A%.txt \
		--system=".venv/bin/python3 ptdriver.py -o ptmodel5A ptdata5A[0-9].txt" \
		+useAnnModel=ptmodel5A.bin +wrDatFile=ptdata5B%.txt \
		--system=".venv/bin/python3 ptdriver.py -o ptmodel5B -i ptmodel5A ptdata5B[0-9].txt" \
		+useAnnModel=ptmodel5B.bin +wrDatFile=ptdata5C%.txt \
		--system=".venv/bin/python3 ptdriver.py -o ptmodel5C -i ptmodel5B ptdata5C[0-9].txt" \
		-exit
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
# Clean

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
