
CC=gcc
CFLAGS=-Wall -Wextra -Werror -Os -MD
DESTDIR=

all: exampleproject

exampleproject: exampleproject.o

install:
	mkdir -p $(DESTDIR)/usr/bin/
	install exampleproject $(DESTDIR)/usr/bin/

clean:
	rm -f exampleproject *.o *.d

-include *.d

