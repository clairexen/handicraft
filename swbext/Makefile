puzzle.out: puzzle.smt2
	yices-smt2 -v 10 puzzle.smt2 > puzzle.tmp
	egrep -A5 -B1 '^sat|))' puzzle.tmp
	mv puzzle.tmp puzzle.out

puzzle.smt2: puzzle.py
	python3 puzzle.py > puzzle.smt2

clean:
	rm -f puzzle.smt2 puzzle.out
