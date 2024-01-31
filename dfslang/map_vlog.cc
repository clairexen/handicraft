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

bool DfsMapSimpleVerilog::map(FILE *f)
{
	fprintf(f, "/* generated using dfslang */\n\n");

	std::string modname = "DFS_CORE";
	if (options.count("modname"))
		modname = options.at("modname");

	bool rename_registers = false;
	if (options.count("noregrename"))
		rename_registers = !atoi(options.at("noregrename").c_str());

	bool use_reg_outputs = true;
	if (options.count("noregout"))
		use_reg_outputs = !atoi(options.at("noregout").c_str());

	fprintf(f, "/***** INSTANCIATION TEMPLATE ******\n");
	fprintf(f, "%s %s (\n", modname.c_str(), modname.c_str());
	fprintf(f, "\t.clk(clk),\n");
	fprintf(f, "\t.rst(rst),\n");
	fprintf(f, "\t.sync_in(sync_in),\n");
	fprintf(f, "\t.sync_out(sync_out)");
	for (auto &it : input_ports)
		fprintf(f, ",\n\t.%s(%s)", it.second.name.c_str(), it.second.name.c_str());
	for (auto &it : output_ports)
		fprintf(f, ",\n\t.%s(%s)", it.second.name.c_str(), it.second.name.c_str());
	fprintf(f, "\n);\n");
	fprintf(f, "***********************************/\n\n");

	fprintf(f, "module %s (\n", modname.c_str());
	fprintf(f, "\t" "input clk,\n");
	fprintf(f, "\t" "input rst,\n");
	fprintf(f, "\t" "output reg sync_in,\n");
	fprintf(f, "\t" "output reg sync_out");
	for (auto &it : input_ports)
		fprintf(f, ",\n\t" "input %s[%d:0] %s", it.second.is_signed ? "signed " : "",
				it.second.bit_width-1, it.second.name.c_str());
	for (auto &it : output_ports)
		fprintf(f, ",\n\t" "output reg %s[%d:0] %s", it.second.is_signed ? "signed " : "",
				it.second.bit_width-1, it.second.name.c_str());
	fprintf(f, "\n);\n\n");

	std::map<std::string, std::string> regname;
	std::map<std::string, int> regnames_counter;

	for (auto &it : registers) {
		char buffer[4096];
		snprintf(buffer, 4096, "\\reg_%s ", packed(it.second.name).c_str());
		if (rename_registers) {
			if (regnames_counter.count(buffer) == 0) {
				int new_id = regnames_counter.size();
				regnames_counter[buffer] = new_id;
			}
			snprintf(buffer, 4096, "reg%03d", regnames_counter[buffer]);
		}
		regname[it.second.name] = buffer;
	}

	for (auto &it : registers) {
		if (packed_registers.count(it.first) > 0)
			continue;
		fprintf(f, "reg %s[%d:0] %s;\n", it.second.is_signed ? "signed " : "",
				it.second.bit_width-1, regname.at(it.second.name).c_str());
	}
	fprintf(f, "\n");

	int state_bits = ceil(log2(max_time+2));
	fprintf(f, "reg [%d:0] state;\n", state_bits-1);
	fprintf(f, "always @(posedge clk) begin\n");
	fprintf(f, "\t" "state <= rst ? %d'd%d : state == %d'd%d ? %d'd0 : state+%d'd1;\n", state_bits, max_time-1, state_bits, max_time-1, state_bits, state_bits);
	fprintf(f, "\t" "sync_in <= !rst && state == %d'd%d;\n", state_bits, max_time-1);
	fprintf(f, "\t" "sync_out <= !rst && state == %d'd%d;\n", state_bits, output_delay);
	fprintf(f, "end\n\n");

	std::map<std::string, std::map<std::string, std::vector<int>>> reg_read_codes;
	for (auto &it : registers) {
		char buffer[4096];
		snprintf(buffer, 4096, "%s <= %s;", regname.at(it.second.name).c_str(), it.second.data_port.c_str());
		reg_read_codes[packed(it.second.name)][buffer].push_back(it.second.data_timing);
	}
	for (auto &it : reg_read_codes)
		if (it.second.size() == 1 && it.second.begin()->second.size() == 1) {
			fprintf(f, "always @(posedge clk)\n" "\tif (state == %d'd%d)\n\t\t%s\n\n",
					state_bits, it.second.begin()->second.at(0), it.second.begin()->first.c_str());
		} else {
			fprintf(f, "always @(posedge clk)\n");
			fprintf(f, "\tcase (state)\n");
			for (auto &it2 : it.second) {
				fprintf(f, "\t\t");
				bool first_item = true;
				for (int t : it2.second) {
					fprintf(f, "%s%d'd%d", first_item ? "" : ", ", state_bits, t);
					first_item = false;
				}
				fprintf(f, ":\n\t\t\t%s\n", it2.first.c_str());
			}
			fprintf(f, "\tendcase\n\n");
		}

	for (auto &it : transactions) {
		bool found_port_to_port_cases = false;
		bool found_const_to_port_cases = false;
		std::string most_common_action, second_most_common_action;
		std::map<std::string, std::set<std::string>> action_consolidation_map;
		for (auto &tr : it.second)
			if (!tr.from_port.empty())
				found_port_to_port_cases = true;
			else if (tr.from_reg.empty())
				found_const_to_port_cases = true;
		for (auto &tr : it.second) {
			char act_code[4096], act_state[4096];
			snprintf(act_state, 4096, "%d'd%d", state_bits, found_port_to_port_cases || use_reg_outputs ? tr.timing : tr.timing+1);
			if (!tr.from_port.empty())
				snprintf(act_code, 4096, "%s <= %s", tr.to_port.c_str(), tr.from_port.c_str());
			else if (!tr.from_reg.empty())
				snprintf(act_code, 4096, "%s <= %s", tr.to_port.c_str(), regname.at(tr.from_reg).c_str());
			else
				snprintf(act_code, 4096, "%s <= %d", tr.to_port.c_str(), tr.constval);
			action_consolidation_map[act_code].insert(act_state);
			if (most_common_action.empty() || action_consolidation_map.at(most_common_action).size() < action_consolidation_map.at(act_code).size())
				most_common_action = act_code;
		}
		if (action_consolidation_map.size() == 1) {
			fprintf(f, "always @%s\n", found_port_to_port_cases || found_const_to_port_cases ? "(posedge clk)" : "*");
			fprintf(f, "\t%s;\n\n", action_consolidation_map.begin()->first.c_str());
		} else {
			for (auto &act : action_consolidation_map) {
				if (act.first == most_common_action)
					continue;
				if (second_most_common_action.empty() || action_consolidation_map.at(second_most_common_action).size() < action_consolidation_map.at(act.first).size())
					second_most_common_action = act.first;
			}
			bool common_as_default = second_most_common_action.empty() || 2*action_consolidation_map.at(second_most_common_action).size()+2 < action_consolidation_map.at(most_common_action).size();
			fprintf(f, "always @%s\n", found_port_to_port_cases || use_reg_outputs ? "(posedge clk)" : "*");
			fprintf(f, "\t" "case (state)\n");
			for (auto &act : action_consolidation_map) {
				if (common_as_default && act.first == most_common_action)
					continue;
				const char *sep = "\t\t";
				for (auto &state : act.second) {
					fprintf(f, "%s%s", sep, state.c_str());
					sep = ", ";
				}
				assert(*sep == ',');
				fprintf(f, ":\n\t\t\t%s;\n", act.first.c_str());
			}
			fprintf(f, "\t\t" "default:\n");
			if (common_as_default)
				fprintf(f, "\t\t\t%s;\n", most_common_action.c_str());
			else
				fprintf(f, "\t\t\t" "%s <= 'bx;\n", it.first.c_str());
			fprintf(f, "\t" "endcase\n\n");
		}
	}

	fprintf(f, "endmodule\n");

	return true;
}

