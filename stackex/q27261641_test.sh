#!/bin/bash

for i in {0..100}; do
	python q27261641_maketest.py > cnf_ksat.dimacs
	python q27261641_convert.py cnf_ksat.dimacs > cnf_3sat.dimacs
	a=$( minisat cnf_ksat.dimacs | grep SATISFIABLE )
	b=$( minisat cnf_3sat.dimacs | grep SATISFIABLE )
	if [ "$a" == "$b" ]; then c="OK"; else c="ERROR"; fi
	printf '%15s %15s %s\n' $a $b $c
done

rm -f cnf_ksat.dimacs
rm -f cnf_3sat.dimacs

