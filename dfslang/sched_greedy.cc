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

namespace
{
	void stamp_reset_core_maps(DfsStamp &s)
	{
		for (auto &map : s.core_maps)
			map = 0;
	}

	bool stamp_next_core_maps(DfsStamp &s)
	{
		for (size_t i = 0; i < s.core_maps.size(); i++) {
			if (++s.core_maps.at(i) < s.cores.at(i)->num_cores)
				return true;
			s.core_maps.at(i) = 0;
		}
		return false;
	}
}

DfsSchedGreedy::DfsSchedGreedy(struct DfsProg *prog) : DfsSched(prog)
{
	next_possible_stamp_map.resize(prog->stamps.size());
}

bool DfsSchedGreedy::schedule()
{
	printf("Running GREEDY scheduler:\n");
	reset();

	Checker checker(this);
	checker.check();

#if 1
	while (1)
	{
		int time_offset, stamp_idx;
		stamp_idx = greedyPick(&time_offset);
		if (stamp_idx < 0)
			break;

		int max_time = max_time_offset;
		if (constraint_max_time_offset.count(stamp_idx))
			max_time = constraint_max_time_offset.at(stamp_idx);

		printf("  scheduling stamp %d:", stamp_idx);
		DfsStamp &s = prog->stamps.at(stamp_idx);

		while (time_offset <= max_time) {
			s.time_offset = time_offset++;
			stamp_reset_core_maps(s);
			do {
				if (checker.check(stamp_idx))
					goto scheduled_stamp;
			} while (stamp_next_core_maps(s));
		}

		printf(" FAILED.\n");
		return false;

	scheduled_stamp:
		printf(" %d", s.time_offset);
		for (int map : s.core_maps)
			printf(" %d", map);
		printf("\n");
	}

	return true;
#else
	while (1)
	{
		int stamp_idx = -1;
		int best_stamp_time = -1;

		for (size_t i = 0; i < prog->stamps.size(); i++)
		{
			auto &s = prog->stamps.at(i);
			if (s.time_offset >= 0)
				continue;
			int this_stamp_time = greedyTry(checker, i, false);
			if (this_stamp_time < 0)
				return false;
			if (stamp_idx < 0 || this_stamp_time < best_stamp_time)
				stamp_idx = i, best_stamp_time = this_stamp_time;
		}

		if (stamp_idx < 0)
			return true;

		best_stamp_time = greedyTry(checker, stamp_idx, true);
		assert(best_stamp_time >= 0);

		printf("  scheduling stamp %d: %d", stamp_idx, prog->stamps.at(stamp_idx).time_offset);
		for (int map : prog->stamps.at(stamp_idx).core_maps)
			printf(" %d", map);
		printf("\n");
	}
#endif
}

int DfsSchedGreedy::greedyTry(Checker &checker, int stamp_idx, bool add_on_success)
{
	DfsStamp &s = prog->stamps.at(stamp_idx);
	int &time_offset = next_possible_stamp_map.at(stamp_idx);

	if (s.time_offset >= 0)
		return s.time_offset;

	int max_time = max_time_offset;
	if (constraint_max_time_offset.count(stamp_idx))
		max_time = constraint_max_time_offset.at(stamp_idx);

	while (time_offset <= max_time) {
		s.time_offset = time_offset++;
		stamp_reset_core_maps(s);
		do {
			if (checker.check(stamp_idx, add_on_success)) {
				if (!add_on_success) {
					s.time_offset = -1;
					stamp_reset_core_maps(s);
				}
				return time_offset;
			}
		} while (stamp_next_core_maps(s));
	}

	s.time_offset = -1;
	stamp_reset_core_maps(s);
	return -1;
}

int DfsSchedGreedy::greedyPick(int *init_time_ptr)
{
	int best_stamp = -1, best_time = -1;

	for (size_t i = 0; i < prog->stamps.size(); i++)
	{
		auto &s = prog->stamps.at(i);
		if (s.time_offset >= 0)
			continue;

		int this_time = 0;
		if (constraint_min_time_offset.count(i))
			this_time = constraint_min_time_offset.at(i);

		for (auto &it : constraint_from_to_delay) {
			if (it.first.second != i)
				continue;
			if (prog->stamps.at(it.first.first).time_offset < 0)
				goto try_next_stamp;
			this_time = std::max(this_time, prog->stamps.at(it.first.first).time_offset + it.second);
		}

		if (this_time < best_time || best_time < 0)
			best_stamp = i, best_time = this_time;
	
	try_next_stamp:;
	}

	if (init_time_ptr != NULL)
		*init_time_ptr = best_time;
	return best_stamp;
}

