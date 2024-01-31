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
#include <set>
#include <assert.h>

void DfsProg::dump()
{
	for (auto &c : cores) {
		printf("\n== CORE: %s (%d) ==\n", c.name.c_str(), c.num_cores);
		for (size_t i = 0; i < c.ports_types.size(); i++)
			printf("Port %s: type=%d, direction=%c\n", c.ports_names.at(i).c_str(), c.ports_types.at(i), c.ports_directions.at(i));
	}

	printf("\n== REGISTERS ==\n");
	for (auto &r : regs) {
		printf("Reg %s %s: type=%d, direction=%c\n", r.name.c_str(), r.index.c_str(), r.type, r.direction);
	}

	for (size_t k = 0; k < stamps.size(); k++) {
		auto &s = stamps.at(k);
		printf("\n== STAMP %d ==\n", int(k));
		printf("Scheduled at time offset %d.\n", s.time_offset);
		for (size_t i = 0; i < s.cores.size(); i++)
			printf("Core %d: %s (mapped to instance %d)\n", int(i), s.cores.at(i)->name.c_str(), s.core_maps.at(i));
		for (size_t i = 0; i < s.regs.size(); i++)
			printf("Reg %d: %s %s\n", int(i), s.regs.at(i)->name.c_str(), s.regs.at(i)->index.c_str());
		for (auto &action : s.actions) {
			if (action.dst_core < 0 && action.dst_port < 0)
				printf("const(%d) <= ", action.dst_arg);
			else if (action.dst_core < 0 && action.dst_arg < 0)
				printf("reg(%s%s) <= ", s.regs.at(action.dst_port)->name.c_str(), s.regs.at(action.dst_port)->index.c_str());
			else if (action.dst_core < 0)
				printf("io(%s%s, %d) <= ", s.regs.at(action.dst_port)->name.c_str(), s.regs.at(action.dst_port)->index.c_str(), action.dst_arg);
			else
				printf("port(%s[%d], %s, %d) <= ", s.cores.at(action.dst_core)->name.c_str(), action.dst_core, s.cores.at(action.dst_core)->ports_names.at(action.dst_port).c_str(), action.dst_arg);
			if (action.src_core < 0 && action.src_port < 0)
				printf("const(%d)\n", action.src_arg);
			else if (action.src_core < 0 && action.src_arg < 0)
				printf("reg(%s%s)\n", s.regs.at(action.src_port)->name.c_str(), s.regs.at(action.src_port)->index.c_str());
			else if (action.src_core < 0)
				printf("io(%s%s, %d)\n", s.regs.at(action.src_port)->name.c_str(), s.regs.at(action.src_port)->index.c_str(), action.src_arg);
			else
				printf("port(%s[%d], %s, %d)\n", s.cores.at(action.src_core)->name.c_str(), action.src_core, s.cores.at(action.src_core)->ports_names.at(action.src_port).c_str(), action.src_arg);
		}
	}

	printf("\n");
}

void DfsProg::optimize()
{
	if (stamps.size() == 0)
		return;
	assert(stamps.size() == 1);

	DfsStamp master = stamps.front();
	master.optimize(false);
	stamps.clear();

	while (!master.actions.empty())
	{
		DfsStamp new_stamp = master;
		new_stamp.actions.clear();

		std::set<int> used_cores;
		bool keep_running = true;

		while (keep_running) {
			keep_running = false;
			std::vector<DfsAction> rem_actions;
			for (auto &act : master.actions) {
				bool use_action = false;
				if (new_stamp.actions.size() == 0)
					use_action = true;
				if (act.dst_core >= 0 && used_cores.count(act.dst_core) > 0)
					use_action = true;
				if (act.src_core >= 0 && used_cores.count(act.src_core) > 0)
					use_action = true;
				if (use_action) {
					if (act.dst_core >= 0)
						used_cores.insert(act.dst_core);
					if (act.src_core >= 0)
						used_cores.insert(act.src_core);
					new_stamp.actions.push_back(act);
					keep_running = true;
				} else {
					rem_actions.push_back(act);
				}
			}
			master.actions.swap(rem_actions);
		}

		new_stamp.optimize(true);
		stamps.push_back(new_stamp);
	}
}

void DfsStamp::optimize(bool init_schedule)
{
	// skip alias assignments

	bool keep_running = true;
	while (keep_running) {
		keep_running = false;
		for (auto act : actions) {
			if (act.dst_core < 0 && act.dst_port >= 0 && act.dst_arg < 0 && act.src_core < 0 && ((act.src_port >= 0 && act.src_arg < 0) || (act.src_port < 0 && act.src_arg >= 0)))
				for (auto &a : actions) {
					if (a.src_core < 0 && a.src_port == act.dst_port && a.src_arg < 0)
						a.src_port = act.src_port, a.src_arg = act.src_arg, keep_running = true;
					if (a.dst_core < 0 && a.dst_port == act.dst_port && a.dst_arg < 0)
						a.dst_port = act.src_port, a.dst_arg = act.src_arg, keep_running = true;
				}
		}
		if (keep_running) {
			std::vector<DfsAction> new_actions;
			for (auto &act : actions)
				if (act.dst_core != act.src_core || act.dst_port != act.src_port || act.dst_arg != act.src_arg)
					new_actions.push_back(act);
			actions.swap(new_actions);
		}
	}

	// remove unused cores and regs from this stamp

	std::set<int> used_cores, used_regs;

	for (auto &act : actions) {
		if (act.dst_core >= 0)
			used_cores.insert(act.dst_core);
		if (act.src_core >= 0)
			used_cores.insert(act.src_core);
		if (act.dst_core < 0 && act.dst_port >= 0)
			used_regs.insert(act.dst_port);
		if (act.src_core < 0 && act.src_port >= 0)
			used_regs.insert(act.src_port);
	}

	std::vector<int> cores_map(cores.size());
	std::vector<DfsCore*> new_cores;

	for (size_t i = 0; i < cores.size(); i++) {
		cores_map[i] = new_cores.size();
		if (used_cores.count(i))
			new_cores.push_back(cores.at(i));
	}

	std::vector<int> regs_map(regs.size());
	std::vector<DfsReg*> new_regs;

	for (size_t i = 0; i < regs.size(); i++) {
		regs_map[i] = new_regs.size();
		if (used_regs.count(i))
			new_regs.push_back(regs.at(i));
	}

	for (auto &act : actions) {
		if (act.dst_core >= 0)
			act.dst_core = cores_map.at(act.dst_core);
		if (act.src_core >= 0)
			act.src_core = cores_map.at(act.src_core);
		if (act.dst_core < 0 && act.dst_port >= 0)
			act.dst_port = regs_map.at(act.dst_port);
		if (act.src_core < 0 && act.src_port >= 0)
			act.src_port = regs_map.at(act.src_port);
	}

	cores.swap(new_cores);
	regs.swap(new_regs);

	// align multi-core timings

	int align_count = 0;
	keep_running = true;
	while (keep_running) {
		keep_running = false;
		assert(align_count++ < 100);
		for (auto &act : actions) {
			if (act.dst_core < 0 || act.src_core < 0)
				continue;
			if (act.src_arg > act.dst_arg) {
				int delta = act.src_arg - act.dst_arg;
				for (auto &a : actions) {
					if (a.dst_core == act.dst_core)
						a.dst_arg += delta;
					if (a.src_core == act.dst_core)
						a.src_arg += delta;
				}
			}
			if (act.dst_arg > act.src_arg) {
				int delta = act.dst_arg - act.src_arg;
				for (auto &a : actions) {
					if (a.dst_core == act.src_core)
						a.dst_arg += delta;
					if (a.src_core == act.src_core)
						a.src_arg += delta;
				}
			}
		}
	}

	if (!init_schedule)
		return;

	// assign core instances in trivial cases

	core_maps.resize(cores.size());
	for (size_t i = 0; i < cores.size(); i++)
		core_maps.at(i) = cores.at(i)->num_cores > 1 ? -1 : 0;

	// assign time offset in trivial cases

	time_offset = -1;
	for (auto &act : actions)
	{
		int abs_time = -1, rel_time = -1;

		if (act.dst_core < 0 && act.dst_port >= 0 && act.dst_arg >= 0)
			abs_time = act.dst_arg;

		if (act.src_core < 0 && act.src_port >= 0 && act.src_arg >= 0)
			abs_time = act.src_arg;

		if (act.dst_core >= 0)
			rel_time = act.dst_arg;

		if (act.src_core >= 0)
			rel_time = act.src_arg;

		if (abs_time >= 0 && time_offset < 0)
			time_offset = abs_time;

		if (abs_time >= 0 && rel_time >= 0) {
			int this_time_offset = abs_time - rel_time;
			assert(this_time_offset >= 0);
			time_offset = this_time_offset;
		}
	}
}

