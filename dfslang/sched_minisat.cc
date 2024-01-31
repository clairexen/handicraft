/*
 *  DfsLang -- Data Flow Scheduling Language
 *
 *  Copyright (C) 2013  RIEGL Research ForschungsGmbH
 *  Copyright (C) 2013  Clifford Wolf <clifford@clifford.at>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#undef VERBOSE
#undef DUMP_DIMACS

#define __STDC_LIMIT_MACROS 1
#include <cinttypes>
#include "minisat/core/Solver.h"

#include "dfslang.h"
#include <assert.h>

namespace
{
	struct stamp_attr_t
	{
		// min and max scheduled time (from constraints)
		int min_time, max_time;

		// one SAT variable for each possible time slot (max_time-min_time+1)
		std::vector<Minisat::Var> time_map_vars;

		// one SAT variable for each core mapping
		std::vector<std::vector<Minisat::Var>> core_map_vars;
	};

	bool sat_add_min_one_hot(Minisat::Solver *solver, std::vector<Minisat::Var> vars)
	{
		Minisat::vec<Minisat::Lit> ps;
		for (auto &v : vars)
			ps.push(Minisat::mkLit(v));
		return solver->addClause(ps);
	}

	bool sat_add_max_one_hot(Minisat::Solver *solver, std::vector<Minisat::Var> vars)
	{
		int num_bits = ceil(log2(vars.size()));

		std::vector<Minisat::Var> bits;
		for (int i = 0; i < num_bits; i++)
			bits.push_back(solver->newVar());

		for (int i = 0; i < int(vars.size()); i++)
		for (int j = 0; j < num_bits; j++)
			if (!solver->addClause(Minisat::mkLit(vars[i], true), Minisat::mkLit(bits[j], (i & (1 << j)) == 0)))
				return false;

		return true;
	}
}

bool DfsSchedMinisat::schedule()
{
	printf("Running SAT scheduler:\n");
	reset();

	Minisat::Solver *solver = new Minisat::Solver;
	std::vector<stamp_attr_t> stamp_attr(prog->stamps.size());

	/////////////////////////////////////////////////////////////////
	// creating model variables and trivial one-hot constraints
	/////////////////////////////////////////////////////////////////

	printf("  Setting up SAT model variables:");
#ifdef VERBOSE
	printf("\n");
#endif
	for (size_t i = 0; i < prog->stamps.size(); i++)
	{
		DfsStamp &s = prog->stamps.at(i);
		stamp_attr_t &sa = stamp_attr.at(i);

		if (s.time_offset < 0) {
			sa.min_time = constraint_min_time_offset.count(i) > 0 ? constraint_min_time_offset.at(i) : 0;
			sa.max_time = constraint_max_time_offset.count(i) > 0 ? constraint_max_time_offset.at(i) : max_time_offset;
		} else {
			sa.min_time = s.time_offset;
			sa.max_time = s.time_offset;
		}

		for (int j = sa.min_time; j <= sa.max_time; j++)
			sa.time_map_vars.push_back(solver->newVar());

		if (sa.min_time < sa.max_time) {
			if (!sat_add_min_one_hot(solver, sa.time_map_vars))
				goto failed;
			if (!sat_add_max_one_hot(solver, sa.time_map_vars))
				goto failed;
		} else {
			if (!solver->addClause(Minisat::mkLit(sa.time_map_vars.back(), false)))
				goto failed;
		}

#ifdef VERBOSE
		printf("    variables for stamp %d time mapping [%d..%d]:", int(i), sa.min_time, sa.max_time);
		for (auto &v : sa.time_map_vars)
			printf("  %d", v);
		printf("\n");
#endif

		for (size_t j = 0; j < s.core_maps.size(); j++) {
			std::vector<Minisat::Var> vars;
			for (int k = 0; k < s.cores[j]->num_cores; k++) {
				vars.push_back(solver->newVar());
				if (s.core_maps[j] >= 0)
					if (!solver->addClause(Minisat::mkLit(vars.back(), k != s.core_maps[j])))
						goto failed;
			}
			if (s.core_maps[j] < 0) {
				if (!sat_add_min_one_hot(solver, vars))
					goto failed;
				if (!sat_add_max_one_hot(solver, vars))
					goto failed;
			}
			sa.core_map_vars.push_back(vars);
		}
	}
#ifdef VERBOSE
	printf("    ...");
#endif
	printf(" %d clauses and %d variables.\n", solver->nClauses(), solver->nVars());

	/////////////////////////////////////////////////////////////////
	// additional constraints from constraint_from_to_delay
	/////////////////////////////////////////////////////////////////

	printf("  Setting up from_to_delay constraints:");
#ifdef VERBOSE
	printf("\n");
#endif
	for (auto &c : constraint_from_to_delay)
	{
		int stamp_a = c.first.first;
		int stamp_b = c.first.second;
		int delay = c.second - 1;

		if (delay < 0) {
			int tmp = stamp_a;
			stamp_a = stamp_b;
			stamp_b = tmp;
			delay = -delay;
		}
#ifdef VERBOSE
		printf("    from %d to %d with delay %d:", stamp_a, stamp_b, delay);
#endif
		stamp_attr_t &sa_a = stamp_attr.at(stamp_a);
		stamp_attr_t &sa_b = stamp_attr.at(stamp_b);

		std::vector<Minisat::Var> vars;

		for (int i = -delay; i <= max_time_offset + delay; i++)
		{
			int a_time = i;
			int b_time = i + delay;

			bool a_out_of_range = a_time < sa_a.min_time || a_time > sa_a.max_time;
			bool b_out_of_range = b_time < sa_b.min_time || b_time > sa_b.max_time;

			if (a_out_of_range && b_out_of_range)
				continue;

			vars.push_back(solver->newVar());

#ifdef VERBOSE
			if (vars.size() == 1)
				printf(" <%d>", vars.back());
			if (vars.size() % 10 == 1)
				printf("\n        ");
			printf(a_out_of_range ? " [(%d)" : " [%d", a_time);
			printf(b_out_of_range ? " (%d)]" : " %d]", b_time);
#endif

			if (vars.size() > 1) {
				Minisat::Var v1 = vars.at(vars.size()-2);
				Minisat::Var v2 = vars.at(vars.size()-1);
				if (!solver->addClause(Minisat::mkLit(v1, true), Minisat::mkLit(v2, false)))
					goto failed;
			}

			if (!a_out_of_range && !solver->addClause(Minisat::mkLit(sa_a.time_map_vars.at(a_time - sa_a.min_time), true), Minisat::mkLit(vars.back(), true)))
				goto failed;
			if (!b_out_of_range && !solver->addClause(Minisat::mkLit(sa_b.time_map_vars.at(b_time - sa_b.min_time), true), Minisat::mkLit(vars.back(), false)))
				goto failed;
		}
#ifdef VERBOSE
		printf("\n");
#endif
	}
#ifdef VERBOSE
	printf("    ...");
#endif
	printf(" %d clauses and %d variables.\n", solver->nClauses(), solver->nVars());

	/////////////////////////////////////////////////////////////////
	// additional constraints from constraint_core_busy
	/////////////////////////////////////////////////////////////////

	printf("  Setting up core_busy constraints:");
	{
		// key fields: core, instance, port, time
		std::map<std::vector<int>, std::vector<Minisat::Var>> core_busy_flags;

		std::map<DfsCore*, int> core_to_id;
		for (size_t i = 0; i < prog->cores.size(); i++)
			core_to_id[&prog->cores[i]] = i;

		for (auto &constr : constraint_core_busy)
		{
			int stamp_id = constr.first.first;
			int stamp_core_id = constr.first.second;

			stamp_attr_t &sa = stamp_attr.at(stamp_id);
			DfsStamp &s = prog->stamps.at(stamp_id);
			DfsCore *c = s.cores.at(stamp_core_id);

			std::vector<int> key;
			key.push_back(core_to_id.at(c));

			for (int i = 0; i < c->num_cores; i++)
			{
				key.push_back(i);
				Minisat::Var core_sel_var = sa.core_map_vars.at(stamp_core_id).at(i);

				for (auto &it : constr.second) {
					key.push_back(it.first);
					key.push_back(it.second + sa.min_time);

					for (int j = sa.min_time; j <= sa.max_time; j++, key.back()++) {
						Minisat::Var time_sel_var = sa.time_map_vars.at(j - sa.min_time);
						Minisat::Var v = solver->newVar();
						if (!solver->addClause(Minisat::mkLit(core_sel_var, true), Minisat::mkLit(time_sel_var, true), Minisat::mkLit(v, false)))
							goto failed;
						core_busy_flags[key].push_back(v);
					}

					key.pop_back();
					key.pop_back();
				}

				key.pop_back();
			}
		}

		for (auto &it : core_busy_flags)
			if (!sat_add_max_one_hot(solver, it.second))
				goto failed;
	}
	printf(" %d clauses and %d variables.\n", solver->nClauses(), solver->nVars());

	/////////////////////////////////////////////////////////////////
	// solving problem and extracting model
	/////////////////////////////////////////////////////////////////

	printf("  Simplifying SAT problem:");
	if (!solver->simplify())
		goto failed;
	printf(" %d clauses and %d variables.\n", solver->nClauses(), solver->nVars());

#ifdef DUMP_DIMACS
	{
		Minisat::vec<Minisat::Lit> empty_assumps;
		solver->toDimacs(stdout, empty_assumps);
	}
#endif

	printf("  Solving SAT problem:");
	fflush(stdout);
	if (!solver->solve())
		goto failed;
	printf(" OK.\n");

	printf("  Extracting model:\n");
#ifdef VERBOSE
	printf("    raw model:");
	for (int i = 0; i < solver->model.size(); i++) {
		if (i % 25 == 0)
			printf("\n        ");
		if (Minisat::toInt(solver->model[i]) == 0)
			printf(" %4d", i);
		else
			printf("  ...");
	}
	printf("\n");
#endif
	for (size_t i = 0; i < prog->stamps.size(); i++)
	{
		DfsStamp &s = prog->stamps.at(i);
		stamp_attr_t &sa = stamp_attr.at(i);

		if (s.time_offset < 0) {
			for (int j = sa.min_time; j <= sa.max_time; j++) {
				Minisat::Var var = sa.time_map_vars.at(j - sa.min_time);
				if (Minisat::toInt(solver->modelValue(var)) == 0) {
					assert(s.time_offset < 0);
					s.time_offset = j;
				}
			}
			assert(s.time_offset >= 0);
			printf("    stamp %d scheduled at %d.\n", int(i), s.time_offset);
		}
	}

	printf("  Checking model:");
	if (!check())
		goto failed;
	printf(" OK.\n");

	delete solver;
	return true;

failed:
	printf(" FAILED.\n");
	delete solver;
	return false;
}

