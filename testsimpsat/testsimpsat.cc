// NOTE: This is not a bug in MiniSat::SimpSolver! A variable most be "frozen"
// using the MiniSat::SimpSolver::setFrozen() API when it will be accessed
// after again (in assumptions or new clauses) after a call to solve().

//
//  Copyright (C) 2014  Clifford Wolf <clifford@clifford.at>
//
//  Permission to use, copy, modify, and/or distribute this software for any
//  purpose with or without fee is hereby granted, provided that the above
//  copyright notice and this permission notice appear in all copies.
//
//  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
//  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
//  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
//  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
//  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//
// ----------------------------------------------------------------------------
//
// There seems to be an issue in Minisat::SimpSolver with incremental SAT solving.
// This program finds a small input pattern that triggers the problem.
//
// Set path to minisat in testsimpsat.sh and run "bash testsimpsat.sh".

#define MAX_VARS 6
#define MAX_ITER 10000
#undef STOP_AT_ERROR

// needed for MiniSAT headers (see Minisat Makefile)
#define __STDC_LIMIT_MACROS
#define __STDC_FORMAT_MACROS

#include <limits.h>
#include <stdint.h>
#include <signal.h>
#include <cinttypes>
#include <algorithm>
#include <utility>

#include <minisat/core/Solver.h>
#include <minisat/simp/SimpSolver.h>

struct Xor128 {
	uint32_t x, y, z, w;
	Xor128() {
		x = 123456789;
		y = 362436069;
		z = 521288629;
		w = 88675123;
	}
	uint32_t operator()() {
		uint32_t t = x ^ (x << 11);
		x = y, y = z, z = w;
		w = w ^ (w >> 19) ^ (t ^ (t >> 8));
		return w & 0xfffff;
	}
} xor128;

bool got_assert;
int count_lines, count_lit;

#ifdef STOP_AT_ERROR
#  define my_assert(_assert_expr_) do { if (_assert_expr_) break; printf("Assert `%s' failed in %s:%d.\n", #_assert_expr_, __FILE__, __LINE__); exit(1); } while (0)
#else
#  define my_assert(_assert_expr_) do { if (_assert_expr_) break; printf("Assert `%s' failed in %s:%d.\n", #_assert_expr_, __FILE__, __LINE__); got_assert = true; } while (0)
#endif

struct SolverTester {
	Minisat::Solver solver_a;
	Minisat::SimpSolver solver_b;

	int newVar() {
		int var_a = solver_a.newVar();
		int var_b = solver_b.newVar();
		my_assert(var_a == var_b);
		return var_a;
	}

	bool addClause(Minisat::vec<Minisat::Lit> &clause) {
		bool ok_a = solver_a.addClause(clause);
		bool ok_b = solver_b.addClause(clause);
		int cursor = printf("  %3d   clause:", ++count_lines);
		count_lit += clause.size();
		for (int i = 0; i < clause.size(); i++)
			cursor += printf(" %2d", (Minisat::var(clause[i])+1) * (Minisat::sign(clause[i]) ? -1 : +1));
		printf("%*s  [%s|%s]\n", 30-cursor, "", (ok_a ? "OK" : "CONFLICT"), (ok_b ? "OK" : "CONFLICT"));
		my_assert(ok_a == ok_b);
		return ok_a && ok_b;
	}

	bool simplify() {
		bool ok_a = solver_a.simplify();
		bool ok_b = solver_b.simplify();
		int cursor = printf("  %3d   simplify: ", ++count_lines);
		printf("%*s  [%s|%s]\n", 30-cursor, "", (ok_a ? "OK" : "CONFLICT"), (ok_b ? "OK" : "CONFLICT"));
		my_assert(ok_a == ok_b);
		return ok_a && ok_b;
	}

	bool solve(Minisat::vec<Minisat::Lit> &assumps) {
		bool ok_a = solver_a.solve(assumps);
		bool ok_b = solver_b.solve(assumps);
		int cursor = printf("  %3d   solve: ", ++count_lines);
		count_lit += assumps.size();
		for (int i = 0; i < assumps.size(); i++)
			cursor += printf(" %2d", (Minisat::var(assumps[i])+1) * (Minisat::sign(assumps[i]) ? -1 : +1));
		printf("%*s  [%s|%s]\n", 30-cursor, "", (ok_a ? "SAT" : "UNSAT"), (ok_b ? "SAT" : "UNSAT"));
		my_assert(ok_a == ok_b);
		return ok_a && ok_b;
	}
};

int main()
{
	Xor128 best_rnd_init;
	int best_count = 10000;

	for (int k = 0; k <= MAX_ITER; k++)
	{
		printf("Running test %d.\n", k);

		if (k == MAX_ITER) {
			printf("<This is a re-run of the shortest test case.>\n");
			xor128 = best_rnd_init;
		}

		printf("    0   init: %08x %08x %08x %08x\n", xor128.x, xor128.y, xor128.z, xor128.w);

		Xor128 this_rnd_init = xor128;
		got_assert = false;
		count_lines = 0;
		count_lit = 0;

		SolverTester tester;
		std::vector<int> vars;
		for (int i = 0; i < MAX_VARS; i++)
			vars.push_back(tester.newVar());

		while (count_lines*10 + count_lit < best_count && !got_assert)
		{
			for (int i = 0, ii = xor128() % 4; i < ii && count_lines*10 + count_lit < best_count; i++) {
				Minisat::vec<Minisat::Lit> ps;
				for (int j = 0, jj = xor128() % 3 + 2; j < jj && j < vars.size(); j++) {
					int p = xor128() % (vars.size() - j);
					ps.push(Minisat::mkLit(vars[p], xor128() % 2));
					std::swap(vars[p], vars[vars.size() - j - 1]);
				}
				if (!tester.addClause(ps))
					goto finished_test;
			}

			if (!tester.simplify())
				goto finished_test;

			for (int i = 0, ii = xor128() % 3 + 1; i < ii && count_lines*10 + count_lit < best_count && !got_assert; i++) {
				Minisat::vec<Minisat::Lit> ps;
				for (int j = 0, jj = xor128() % 4; j < jj && j < vars.size(); j++) {
					int p = xor128() % (vars.size() - j);
					ps.push(Minisat::mkLit(vars[p], xor128() % 2));
					std::swap(vars[p], vars[vars.size() - j - 1]);
				}
				tester.solve(ps);
			}
		}

	finished_test:;
		if (got_assert && count_lines*10 + count_lit < best_count) {
			best_rnd_init = this_rnd_init;
			best_count = count_lines*10 + count_lit;
		}
	}

	return 0;
}

