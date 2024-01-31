
CC = clang
CFLAGS += -Wall -Os -ggdb -std=gnu99 -MD
LDFLAGS +=
LDLIBS += -lSDL -lm

beatplot: beatplot.o beatlock.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

test: beatplot demo.dat
	./beatplot < demo.dat

demo.dat: demo.mp3
	ffmpeg -y -i demo.mp3 demo.wav
	sox demo.wav -b8 -c1 -r22050 -tu8 demo.dat

clean:
	rm -f beatplot *.o *.d
	rm -f demo.dat

-include *.d

