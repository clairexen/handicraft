run: run-cvc5 run-z3

run-cvc5:
	cvc5 --lang smt --produce-models counter.smt26

run-z3:
	z3 -smt2 counter.smt26
