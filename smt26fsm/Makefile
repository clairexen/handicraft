run: run-recursion run-recursion2 run-assertstate


# Recursive Functions

run-recursion: run-recursion-cvc5 run-recursion-z3

run-recursion-cvc5:
	cvc5 --produce-models --incremental recursion.smt2 

run-recursion-z3:
	z3 recursion.smt2 


# Recursive Functions, Mk2

run-recursion2: run-recursion2-cvc5 run-recursion2-z3

run-recursion2-cvc5:
	-cvc5 --tlimit=2000 --produce-models --incremental recursion2.smt2 

run-recursion2-z3:
	z3 -T:2 recursion2.smt2 


# Assert State Relationships

run-assertstate: run-assertstate-cvc5 run-assertstate-z3

run-assertstate-cvc5:
	cvc5 --produce-models --incremental assertstate.smt2 

run-assertstate-z3:
	z3 assertstate.smt2 

