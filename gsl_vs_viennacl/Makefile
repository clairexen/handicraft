
CC = gcc
CFLAGS += -fopenmp -O3 -Wall -MD -D__STDC_LIMIT_MACROS
LDLIBS += -fopenmp -lm -lstdc++

CFLAGS += $(shell gsl-config --cflags)
LDLIBS += $(shell gsl-config --libs)

CFLAGS += -I/usr/local/ViennaCL-1.3.0
LDLIBS += -lOpenCL

demo: main.o solve_gsl.o solve_viennacl.o
	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS)

%.o: %.cc
	$(CC) -c $(CFLAGS) $<

clean:
	rm -f demo *.o *.d matrix.txt vectors.txt

-include *.d

