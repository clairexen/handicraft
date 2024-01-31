
CFLAGS += -Wall -Wextra -ggdb -std=gnu99
LDFLAGS += -ggdb
LDLIBS += -lppm -lpbm -lm
# LDLIBS += -lnetpbm -lm

brdtrace: brdtrace.o

test: brdtrace
	bash brd2ngc.sh -i example.brd
	# mv brd2ngc.out/overview.png example_out.png

install: brdtrace
	install brdtrace /usr/local/bin/brdtrace
	install brd2ngc.sh /usr/local/bin/brd2ngc

uninstall:
	rm -f /usr/local/bin/brdtrace
	rm -f /usr/local/bin/brd2ngc

clean:
	rm -rf brdtrace brdtrace.o brd2ngc.out example_new.png

