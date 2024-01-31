
CC = gcc
CFLAGS = -Wall -Wextra -Os

all: rdf2ascii ascii2rdf
rdf2ascii: rdf2ascii.o
ascii2rdf: ascii2rdf.o

install: rdf2ascii ascii2rdf
	sudo install rdf2ascii ascii2rdf /usr/local/bin/

clean:
	rm -f rdf2ascii ascii2rdf *.o

