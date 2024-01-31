// NOTE: This is not a bug in MiniSat::SimpSolver! A variable most be "frozen"
// using the MiniSat::SimpSolver::setFrozen() API when it will be accessed
// after again (in assumptions or new clauses) after a call to solve().

//  Copyright (C) 2014  Clifford Wolf <clifford@clifford.at>
//
//  Permission to use, copy, modify, and/or distribute this software for any
//  purpose with or without fee is hereby granted, provided that the above
//  copyright notice and this permission notice appear in all copies.

// needed for MiniSAT headers (see Minisat Makefile)
#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

#include <stdio.h>
#include <minisat/core/Solver.h>
#include <minisat/simp/SimpSolver.h>

using namespace Minisat;

template<typename Solver>
void testcase()
{
	Solver solver;

	Lit v1 = mkLit(solver.newVar());
	Lit v2 = mkLit(solver.newVar());

	vec<Lit> clause;
	clause.push(v1);
	clause.push(v2);

	vec<Lit> assump;
	assump.push(~v1);
	assump.push(~v2);

	printf(" %d", solver.addClause(clause));
	printf(" %d", solver.simplify());

#ifndef SKIP_FIRST
	if (solver.solve())
		printf(" 1<%d%d>", toInt(solver.modelValue(v1)), toInt(solver.modelValue(v2)));
	else
		printf(" 0<-->");
#endif

#ifdef TWO_CLAUSES
	printf(" %d", solver.addClause(assump));

	if (solver.solve())
		printf(" 1<%d%d>", toInt(solver.modelValue(v1)), toInt(solver.modelValue(v2)));
	else
		printf(" 0<-->");
#else
	if (solver.solve(assump))
		printf(" 1<%d%d>", toInt(solver.modelValue(v1)), toInt(solver.modelValue(v2)));
	else
		printf(" 0<-->");
#endif
}

int main()
{
	printf("Minisat::Solver     -> ");
	testcase<Solver>();
	printf("\n");

	printf("Minisat::SimpSolver -> ");
	testcase<SimpSolver>();
	printf("\n");

	return 0;
}

