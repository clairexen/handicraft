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

#ifndef DFSLANG_H
#define DFSLANG_H

#include <map>
#include <set>
#include <vector>
#include <string>
#include <stdio.h>

struct DfsCore
{
	int num_cores;
	std::string name;
	std::vector<int> ports_types;
	std::vector<char> ports_directions;
	std::vector<std::string> ports_names;
	std::map<std::string, int> ports_map;
};

struct DfsReg
{
	std::string name, index;
	char direction; // 'i', 'o' or 'v'
	int type;
};

struct DfsAction
{
	// Encoding of constants:
	//   core = -1, port = -1, arg = <value>
	//
	// Encoding of registers:
	//   core = -1, port = <reg idx>, arg = -1
	//
	// Encoding of ports:
	//   core = -1, port = <reg idx>, arg = <time>
	//
	// Encoding of core ports:
	//   core = <core idx>, port = <port idx>, arg = <time>

	int dst_core, dst_port, dst_arg;
	int src_core, src_port, src_arg;
};

struct DfsStamp
{
	// scheduling
	int time_offset;
	std::vector<int> core_maps;

	// input data
	std::vector<DfsCore*> cores;
	std::vector<DfsReg*> regs;
	std::vector<DfsAction> actions;

	void optimize(bool init_schedule);
};

struct DfsProg
{
	int output_delay;
	std::string set_scheduler;
	std::map<std::string, std::string> options;

	std::vector<DfsCore> cores;
	std::vector<DfsReg> regs;
	std::vector<DfsStamp> stamps;

	void dump();
	void optimize();
};

struct DfsSched
{
	struct DfsProg *prog;
	std::vector<int*> all_variables;
	int max_time_offset;

	// constraint: prog->stamps[it.first.second].time_offset - prog->stamps[it.first.first].time_offset > it.second
	//
	//	it.first.first ..... stamp providing the variable
	//	it.first.second .... stamp depending on the variable
	//	it.second .......... minimum time difference between stamps
	//
	std::map<std::pair<int, int>, int> constraint_from_to_delay;

	// constraint: prog->stamps[it.first].time_offset [ <= or >= ] it.second
	//
	//	it.first ...... contrained stamp
	//	it.second ..... min or max value for .time_offset
	//
	std::map<int, int> constraint_min_time_offset, constraint_max_time_offset;

	// constraint: { prog->stamps[it.first.first].cores[it.first.second]], prog->stamps[it.first.first].core_maps[it.first.second],
	//               it.second...first, (prog->stamps[it.first.first].time_offset + it.second...second) } is unique
	//
	//	it.first.first ........ stamp accessing the port
	//	it.first.second ....... core index within stamp
	//	it.second...first ..... port index on that core
	//	it.second...second .... relative time of port access
	//
	std::map<std::pair<int, int>, std::set<std::pair<int, int>>> constraint_core_busy;

	DfsSched(struct DfsProg *prog);
	virtual ~DfsSched() {};
	virtual bool schedule();

	void reset();
	bool check();
	void dump();

	struct Checker {
		DfsSched *sched;
		std::map<std::pair<DfsCore*, int>, std::set<std::pair<int, int>>> used_core_port;
		std::map<int, std::set<std::pair<int, int>>> constraint_from_to_delay_cache;
		std::map<int, std::set<std::pair<int, int>>> constraint_core_busy_cache;
		bool check(int stamp_idx = -1, bool add_on_success = true);
		Checker(DfsSched *sched);
	};
};

struct DfsSchedGreedy : DfsSched
{
	std::vector<int> next_possible_stamp_map;

	DfsSchedGreedy(struct DfsProg *prog);
	virtual bool schedule();
	int greedyTry(Checker &checker, int stamp_idx, bool add_on_success);
	int greedyPick(int *init_time_ptr = NULL);
};

struct DfsSchedMinisat : DfsSched
{
	DfsSchedMinisat(struct DfsProg *prog) : DfsSched(prog) {}
	virtual bool schedule();
};

struct DfsMap
{
	std::map<std::string, std::string> options;

	struct port_t {
		std::string name;
		int bit_width;
		bool is_signed;
	};

	struct register_t : port_t {
		std::string data_port;
		int data_timing;
	};

	struct transaction_t {
		std::string from_port, from_reg, to_port;
		int timing, constval;
	};

	int max_time, output_delay;
	std::map<std::string, port_t> input_ports, output_ports;
	std::map<std::string, register_t> registers;
	std::map<std::string, std::vector<transaction_t>> transactions;
	std::map<std::string, std::string> packed_registers;

	DfsMap(DfsProg *prog);
	virtual ~DfsMap() {};
	virtual bool map(FILE *f);

	std::string packed(std::string name);
	void pack_greedy();
};

struct DfsMapSimpleVerilog : DfsMap
{
	DfsMapSimpleVerilog(struct DfsProg *prog) : DfsMap(prog) {}
	virtual bool map(FILE *f);
};

struct AstNode
{
	int value;
	std::string type, str;
	std::vector<AstNode*> children;

	AstNode(AstNode &other, std::string suffix = std::string());
	AstNode(std::string type, int value, AstNode *child1 = NULL, AstNode *child2 = NULL, AstNode *child3 = NULL);
	AstNode(std::string type, std::string str, AstNode *child1 = NULL, AstNode *child2 = NULL, AstNode *child3 = NULL);
	AstNode(std::string type, AstNode *child1 = NULL, AstNode *child2 = NULL, AstNode *child3 = NULL);
	~AstNode();

	void dump_code(std::string indent = std::string());
	void dump(std::string indent = std::string());
	void make_prog(DfsProg &prog);
	void optimize();

	void map_reg(DfsStamp &stamp, std::map<std::string, int> &cores_map, std::map<std::pair<std::string, std::string>, int> &regs_map, int &reg_core, int &reg_port, int &reg_arg);
	void optimize_cleanup();
	bool optimize_worker(AstNode *parent, size_t parent_idx, std::vector<AstNode*> &parents, bool do_call);
	AstNode *specialize(std::string varname, int value);
	void expand_stmts(std::vector<AstNode*> &new_children);
	AstNode *instanciate(std::map<std::string, AstNode*> rename_map, std::string rename_suffix, std::vector<AstNode*> &mempool);
	void delete_children();
};

extern AstNode *root;
extern int yydebug;
extern int yylex(void);
extern int yyget_lineno(void);
extern int yyparse(void);
extern int yylex_destroy(void);
extern void yyrestart(FILE*);

#endif
