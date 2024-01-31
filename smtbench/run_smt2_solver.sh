#!/bin/bash
#
# Usage example:
# bash run_smt2_solver.sh all 4h ponylink_maxtx

run_solver() {
	echo "###################"
	echo "SOLVER     $1"
	echo "TIMEOUT    $2"
	echo "BENCHMARK  $3"
	case "$1" in
		z3) timeout $2 z3 -smt2 -in < "$3.smt2" ;;
		cvc4) timeout $2 cvc4 --incremental --lang smt2 < "$3.smt2" ;;
		mathsat) timeout $2 mathsat < "$3.smt2" ;;
		yices) timeout $2 yices-smt2 --incremental < "$3.smt2" ;;
	esac | awk 'BEGIN { x = systime(); } { print strftime("%H:%M:%S", systime()-x, 1), NR, $0; fflush(); }' | tee "$3.out_$1"
}

if [ "$1" == "all" ]; then
	run_solver z3      "$2" "$3"
	run_solver cvc4    "$2" "$3"
	run_solver mathsat "$2" "$3"
	run_solver yices   "$2" "$3"
	exit 0
else
	run_solver "$1" "$2" "$3"
fi

