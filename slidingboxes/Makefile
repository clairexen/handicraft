
TRG += example01.svg
TRG += example02.svg
TRG += example03.svg
TRG += example04.svg
TRG += example05.svg
TRG += example06.svg
TRG += example07.svg

all: $(TRG)

example%.svg: example%.txt slidingboxes
	./slidingboxes < $< | dot -Tsvg -o $@

slidingboxes: slidingboxes.c
	gcc -O3 -march=native -o slidingboxes -Wall slidingboxes.c

clean:
	rm -f slidingboxes *.svg

