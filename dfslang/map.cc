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
#include <math.h>

DfsMap::DfsMap(DfsProg *prog)
{
	output_delay = prog->output_delay;
	max_time = prog->output_delay+2;
	options = prog->options;

	for (auto &s : prog->stamps)
	for (auto &act : s.actions) {
		int abs_time = -1;
		if (act.dst_core < 0 && act.dst_port >= 0 && act.dst_arg >= 0)
			abs_time = act.dst_arg;
		if (act.src_core < 0 && act.src_port >= 0 && act.src_arg >= 0)
			abs_time = act.src_arg;
		max_time = std::max(max_time, abs_time + 1);
	}

	std::set<std::string> unused_regs;

	for (auto &r : prog->regs) {
		register_t reg;
		reg.name = r.name + r.index;
		reg.bit_width = abs(r.type);
		reg.is_signed = r.type < 0;
		reg.data_port = std::string();
		reg.data_timing = -1;
		if (r.direction == 'i')
			input_ports[reg.name] = reg;
		else if (r.direction == 'o')
			output_ports[reg.name] = reg;
		else {
			registers[reg.name] = reg;
			unused_regs.insert(reg.name);
		}
	}

	for (auto &c : prog->cores)
	for (int i = 0; i < c.num_cores; i++)
	for (size_t j = 0; j < c.ports_types.size(); j++) {
		char buffer[4096];
		snprintf(buffer, 4096, "%s_%d_%s", c.name.c_str(), i, c.ports_names.at(j).c_str());
		port_t port;
		port.name = buffer;
		port.bit_width = abs(c.ports_types.at(j));
		port.is_signed = c.ports_types.at(j) < 0;
		if (c.ports_directions.at(j) == 'i')
			output_ports[port.name] = port;
		else
			input_ports[port.name] = port;
	}

	for (auto &s : prog->stamps)
	for (auto &act : s.actions)
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

		if (rel_time >= 0) {
			assert(abs_time < 0 || abs_time == rel_time + s.time_offset);
			abs_time = rel_time + s.time_offset;
		}

		if (abs_time < 0 && rel_time < 0)
			abs_time = s.time_offset;

		bool to_reg = false;
		transaction_t tr;
		tr.timing = abs_time;
		tr.constval = 0;

		if (act.src_core < 0 && act.src_port < 0) {
			tr.constval = act.src_arg;
		} else if (act.src_core < 0 && act.src_arg < 0) {
			tr.from_reg = s.regs.at(act.src_port)->name + s.regs.at(act.src_port)->index;
			unused_regs.erase(tr.from_reg);
		} else if (act.src_core < 0) {
			tr.from_port = s.regs.at(act.src_port)->name.c_str();
		} else {
			char buffer[4096];
			snprintf(buffer, 4096, "%s_%d_%s", s.cores.at(act.src_core)->name.c_str(), s.core_maps.at(act.src_core), s.cores.at(act.src_core)->ports_names.at(act.src_port).c_str());
			tr.from_port = buffer;
		}

		if (act.dst_core < 0 && act.dst_port < 0) {
			assert(!"Constant is an invalid transaction target!");
		} else if (act.dst_core < 0 && act.dst_arg < 0) {
			to_reg = true;
			tr.to_port = s.regs.at(act.dst_port)->name + s.regs.at(act.dst_port)->index;
		} else if (act.dst_core < 0) {
			tr.to_port = s.regs.at(act.dst_port)->name.c_str();
		} else {
			char buffer[4096];
			snprintf(buffer, 4096, "%s_%d_%s", s.cores.at(act.dst_core)->name.c_str(), s.core_maps.at(act.dst_core), s.cores.at(act.dst_core)->ports_names.at(act.dst_port).c_str());
			tr.to_port = buffer;
		}

		if (to_reg) {
			assert(!tr.from_port.empty());
			assert(registers.count(tr.to_port));
			registers.at(tr.to_port).data_port = tr.from_port;
			registers.at(tr.to_port).data_timing = tr.timing;
		} else
			transactions[tr.to_port].push_back(tr);
	}

	for (auto &name : unused_regs)
		registers.erase(name);
}

bool DfsMap::map(FILE *f)
{
	fprintf(f, ".max_time %d\n", max_time);

	for (auto &it : input_ports)
		fprintf(f, ".input %d %c %s\n", it.second.bit_width, it.second.is_signed ? 's' : 'u', it.second.name.c_str());

	for (auto &it : output_ports)
		fprintf(f, ".output %d %c %s\n", it.second.bit_width, it.second.is_signed ? 's' : 'u', it.second.name.c_str());

	for (auto &it : registers)
		fprintf(f, ".reg %d %c %d %s %s\n", it.second.bit_width, it.second.is_signed ? 's' : 'u',
				it.second.data_timing, it.second.data_port.c_str(), it.second.name.c_str());

	for (auto &it : transactions)
	for (auto &tr : it.second)
		if (!tr.from_port.empty())
			fprintf(f, ".tr_port %d %s %s\n", tr.timing, tr.from_port.c_str(), tr.to_port.c_str());
		else if (!tr.from_reg.empty())
			fprintf(f, ".tr_reg %d %s %s\n", tr.timing, tr.from_reg.c_str(), tr.to_port.c_str());
		else
			fprintf(f, ".tr_const %d %d %s\n", tr.timing, tr.constval, tr.to_port.c_str());

	for (auto &it : packed_registers)
		fprintf(f, ".pack %s %s\n", it.first.c_str(), it.second.c_str());

	return true;
}

static bool int_ranges_intersect(int a1, int a2, int b1, int b2)
{
	if (a2 < b1 || b2 < a1)
		return false;
	return true;
}

std::string DfsMap::packed(std::string name)
{
	if (packed_registers.count(name) > 0)
		return packed_registers.at(name);
	return name;
}

void DfsMap::pack_greedy()
{
	std::map<std::string, int> last_reg_access;

	for (auto &it : transactions)
	for (auto &tr : it.second)
		if (!tr.from_reg.empty())
			last_reg_access[tr.from_reg] = std::max(last_reg_access[tr.from_reg], tr.timing);

	for (auto &it : registers) {
		std::set<std::string> pack_candidates;
		assert(last_reg_access.count(it.first) != 0);
		for (auto &it2 : registers) {
			if (it.first == it2.first)
				break;
			if (packed_registers.count(it2.first))
				continue;
			if (it.second.bit_width == it2.second.bit_width && it.second.is_signed == it2.second.is_signed)
				pack_candidates.insert(it2.first);
		}
		int a1 = it.second.data_timing;
		int a2 = last_reg_access.at(it.first);
		for (auto &it2 : registers) {
			if (it.first == it2.first)
				break;
			int b1 = it2.second.data_timing;
			int b2 = last_reg_access.at(it2.first);
			if (!int_ranges_intersect(a1, a2, b1, b2))
				continue;
			if (packed_registers.count(it2.first))
				pack_candidates.erase(packed_registers.at(it2.first));
			else
				pack_candidates.erase(it2.first);
		}
		if (pack_candidates.size() > 0)
			packed_registers[it.first] = *pack_candidates.begin();
	}
}

