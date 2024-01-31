
CXX = clang
CXXFLAGS = -std=c++11 -Wall -Wextra -Os -ggdb
LDLIBS = -lstdc++ -lm

all: avsync_video avsync_audio

avsync_video: LDLIBS += $(shell pkg-config opencv --libs)
avsync_video: aruco-1.2.4/src/libaruco.a avsync_video.o
avsync_video.o: CXXFLAGS += -Iaruco-1.2.4/src/

avsync_audio: avsync_audio.o

markers.js: aruco-1.2.4/src/libaruco.a genmarkers.py
	python genmarkers.py > markers.js

aruco-1.2.4/src/libaruco.a:
	cd aruco-1.2.4 && cmake -DBUILD_SHARED_LIBS=OFF . && make

