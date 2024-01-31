
demo: demo_c.o demo_f.o
	gcc -o $@ $^ -lgfortran -lm -ggdb

demo_f.o: demo_f.f90
	gfortran -c -ggdb -o $@ $<

demo_c.o: demo_c.c
	gcc -c -ggdb -o $@ $<

