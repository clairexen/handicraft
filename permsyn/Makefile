
TEST_PATTERN=v-3-8-19ba--2-7--1--efghij-rpq-0

run: permsyn
	./permsyn $(TEST_PATTERN)

run_g: permsyn
	./permsyn -g $(TEST_PATTERN)

run_b: permsyn
	./permsyn -b $(TEST_PATTERN)

run_gb: permsyn
	./permsyn -g -b $(TEST_PATTERN)

test: permsyn
	python3 gentest.py | tee tst.c
	clang -Wall -O2 -o tst tst.c
	./tst
	rm -f tst.c tst

permsyn: permsyn.cc
	clang++ -Wall -O2 -std=c++11 -o permsyn permsyn.cc

