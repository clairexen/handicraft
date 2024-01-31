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

#include "dfslang.h"
#include <assert.h>
#include <time.h>

DfsSched::DfsSched(struct DfsProg *prog) : prog(prog)
{
	// find all variables

	for (auto &s : prog->stamps) {
		if (s.time_offset < 0)
			all_variables.push_back(&s.time_offset);
		for (auto &cm : s.core_maps)
			if (cm < 0)
				all_variables.push_back(&cm);
	}

	// find global maximum time

	max_time_offset = 0;
	for (auto &s : prog->stamps)
	for (auto &act : s.actions) {
		int abs_time = -1;
		if (act.dst_core < 0 && act.dst_port >= 0 && act.dst_arg >= 0)
			abs_time = act.dst_arg;
		if (act.src_core < 0 && act.src_port >= 0 && act.src_arg >= 0)
			abs_time = act.src_arg;
		max_time_offset = std::max(max_time_offset, abs_time+1);
	}

	// find stamp-to-stamp delay constraints:
	//	constraint_from_to_delay, constraint_min_time_offset, constraint_max_time_offset

	std::map<DfsReg*, std::pair<DfsStamp*, int>> reg_provided_by_at;
	std::map<DfsReg*, int> reg_provided_at;
	std::map<DfsStamp*, int> stamp_to_idx;

	for (size_t i = 0; i < prog->stamps.size(); i++)
		stamp_to_idx[&prog->stamps[i]] = i;

	for (auto &s : prog->stamps)
	for (auto &a : s.actions)
		if (a.dst_core < 0 && a.dst_port >= 0 && a.dst_arg < 0) {
			assert(a.src_port >= 0 && a.src_arg >= 0);
			if(s.regs.at(a.dst_port)->direction != 'o') {
				assert(reg_provided_by_at.count(s.regs.at(a.dst_port)) == 0);
				assert(reg_provided_at.count(s.regs.at(a.dst_port)) == 0);
				if (a.src_core < 0 && s.regs.at(a.src_port)->direction == 'i')
					reg_provided_at[s.regs.at(a.dst_port)] = a.src_arg;
				else
					reg_provided_by_at[s.regs.at(a.dst_port)] = std::pair<DfsStamp*, int>(&s, a.src_arg);
			}
		}

	for (auto &s : prog->stamps)
	for (auto &a : s.actions)
		if (a.src_core < 0 && a.src_port >= 0 && a.src_arg < 0) {
			assert(a.dst_port >= 0 && a.dst_arg >= 0);
			if (s.regs.at(a.src_port)->direction != 'i') {
				if (a.dst_core < 0 && s.regs.at(a.dst_port)->direction == 'o') {
					if (reg_provided_by_at.count(s.regs.at(a.src_port)) == 0)
						continue;
					DfsStamp *ps = reg_provided_by_at[s.regs.at(a.src_port)].first;
					int max_time = a.dst_arg - reg_provided_by_at[s.regs.at(a.src_port)].second - 1;
					if (constraint_max_time_offset.count(stamp_to_idx.at(ps)) > 0)
						max_time = std::min(max_time, constraint_max_time_offset.at(stamp_to_idx.at(ps)));
					constraint_max_time_offset[stamp_to_idx.at(ps)] = max_time;
					// printf("constraint_max_time_offset: %5d %5d         (%s%s)\n", stamp_to_idx.at(ps), max_time,
					// 		s.regs.at(a.src_port)->name.c_str(), s.regs.at(a.src_port)->index.c_str());
				} else
				if (reg_provided_by_at.count(s.regs.at(a.src_port)) != 0) {
					int min_time_delta = reg_provided_by_at[s.regs.at(a.src_port)].second - a.dst_arg + 1;
					std::pair<int, int> key(stamp_to_idx.at(reg_provided_by_at[s.regs.at(a.src_port)].first), stamp_to_idx.at(&s));
					if (constraint_from_to_delay.count(key) > 0)
						min_time_delta = std::max(min_time_delta, constraint_from_to_delay.at(key));
					constraint_from_to_delay[key] = min_time_delta;
					// printf("constraint_from_to_delay:   %5d %5d %5d   (%s%s)\n", key.first, key.second, min_time_delta,
					// 		s.regs.at(a.src_port)->name.c_str(), s.regs.at(a.src_port)->index.c_str());
				} else
				if (reg_provided_at.count(s.regs.at(a.src_port)) != 0) {
					int min_time = reg_provided_at[s.regs.at(a.src_port)] - a.dst_arg + 1;
					if (constraint_min_time_offset.count(stamp_to_idx.at(&s)) > 0)
						min_time = std::max(min_time, constraint_min_time_offset.at(stamp_to_idx.at(&s)));
					constraint_min_time_offset[stamp_to_idx.at(&s)] = min_time;
					// printf("constraint_min_time_offset: %5d %5d         (%s%s)\n", stamp_to_idx.at(&s), min_time,
					// 		s.regs.at(a.src_port)->name.c_str(), s.regs.at(a.src_port)->index.c_str());
				} else
					assert(!"No driver found for signal during dependency analysis!");
			}
		}

	// find core arbitration constraints:
	//	constraint_core_busy

	for (auto &s : prog->stamps)
	for (auto &a : s.actions)
		if (a.dst_core >= 0) {
			std::pair<int, int> key(stamp_to_idx.at(&s), a.dst_core);
			std::pair<int, int> entry(a.dst_port, a.dst_arg);
			constraint_core_busy[key].insert(entry);
		}
}

bool DfsSched::schedule()
{
	static int call_counter = 0;
	uint64_t seed_value = (uint64_t(time(NULL)) << 16) | (uint64_t(call_counter++) << 32) | getpid();

	printf("Running random scheduler with seed %llu:\n", (long long)seed_value);
	printf("WARNING: Use a real scheduler! This is just a dummy for testing the basic constraints.\n");

	srand48(seed_value);
	for (int iter = 0; iter < 1000; iter++) {
		reset();
		printf("  Trying random values:");
		for (auto &s : prog->stamps) {
			if (s.time_offset < 0) {
				s.time_offset = lrand48() % max_time_offset;
				printf(" %3d", s.time_offset);
			}
			for (size_t i = 0; i < s.core_maps.size(); i++)
				if (s.core_maps[i] < 0) {
					s.core_maps[i] = lrand48() % s.cores[i]->num_cores;
					printf(" %3d", s.core_maps[i]);
				}
		}
		printf("\n");
		if (check()) {
			printf("Found solution!\n");
			return true;
		}
	}

	printf("Giving up!\n");
	return false;
}

void DfsSched::reset()
{
	for (auto &it : all_variables)
		*it = -1;
}

bool DfsSched::check()
{
	Checker checker(this);
	return checker.check();
}

DfsSched::Checker::Checker(DfsSched *sched) : sched(sched)
{
	for (auto &it : sched->constraint_from_to_delay) {
		constraint_from_to_delay_cache[it.first.first].insert(it.first);
		constraint_from_to_delay_cache[it.first.second].insert(it.first);
	}

	for (auto &it : sched->constraint_core_busy)
		constraint_core_busy_cache[it.first.first].insert(it.first);
}

bool DfsSched::Checker::check(int stamp_idx, bool add_on_success)
{
	if (stamp_idx < 0) {
		for (size_t i = 0; i < sched->prog->stamps.size(); i++)
			if (sched->prog->stamps.at(i).time_offset >= 0 && !check(i, add_on_success))
				return false;
		return true;
	}

	DfsStamp &s = sched->prog->stamps.at(stamp_idx);
	if (s.time_offset < 0)
		return true;

	for (auto &key : constraint_from_to_delay_cache[stamp_idx]) {
		if (sched->prog->stamps.at(key.first).time_offset < 0)
			continue;
		if (sched->prog->stamps.at(key.second).time_offset < 0)
			continue;
		if (sched->prog->stamps.at(key.second).time_offset - sched->prog->stamps.at(key.first).time_offset < sched->constraint_from_to_delay.at(key))
			return false;
	}

	if (sched->constraint_min_time_offset.count(stamp_idx))
		if (s.time_offset < sched->constraint_min_time_offset.at(stamp_idx))
			return false;

	if (sched->constraint_max_time_offset.count(stamp_idx))
		if (s.time_offset > sched->constraint_max_time_offset.at(stamp_idx))
			return false;

	std::map<std::pair<DfsCore*, int>, std::set<std::pair<int, int>>> this_used_core_port;

	for (auto &key : constraint_core_busy_cache[stamp_idx]) {
		std::set<std::pair<int, int>> &acc = sched->constraint_core_busy.at(key);
		std::pair<DfsCore*, int> ukey(s.cores.at(key.second), s.core_maps.at(key.second));
		auto &this_usage_details = this_used_core_port[ukey];
		auto &usage_details = used_core_port[ukey];
		for (auto entry : acc) {
			entry.second += s.time_offset;
			if (usage_details.count(entry) > 0 || this_usage_details.count(entry))
				return false;
			this_usage_details.insert(entry);
		}
	}

	if (add_on_success)
		for (auto &it : this_used_core_port)
			used_core_port[it.first].insert(it.second.begin(), it.second.end());

	return true;
}

void DfsSched::dump()
{
	printf("----------------\n");
	printf("scheduling problem with %d variables.\n", int(all_variables.size()));
	printf("constraint_from_to_delay:\n");
	for (auto &it : constraint_from_to_delay)
		printf("%5d %5d %5d\n", it.first.first, it.first.second, it.second);
	printf("constraint_min_time_offset:\n");
	for (auto &it : constraint_min_time_offset)
		printf("%5d %5d\n", it.first, it.second);
	printf("constraint_max_time_offset:\n");
	for (auto &it : constraint_max_time_offset)
		printf("%5d %5d\n", it.first, it.second);
	printf("constraint_core_busy:\n");
	for (auto &it : constraint_core_busy) {
		printf("%5d %5d  [", it.first.first, it.first.second);
		const char *spacer = "";
		for (auto &it2 : it.second)
			printf("%s%d:%d", spacer, it2.first, it2.second), spacer = ", ";
		printf("]\n");
	}
	printf("----------------\n");
}

