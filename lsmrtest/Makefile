
lsmrtest: lsmrtest.c lsmr/lsmrModule.o
	gcc -o lsmrtest lsmrtest.c lsmr/lsmrModule.o lsmr/lsmrblas.o -lgfortran -lm -ggdb

lsmr/lsmrModule.o:
	make -C lsmr

clean:
	make -C lsmr clean
	rm -f lsmrtest lsmr/TestProgram

