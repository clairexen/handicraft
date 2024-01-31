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
#include <errno.h>
#include <string.h>

AstNode *root = NULL;

int main(int argc, char **argv)
{
	FILE *f;
	std::string prefix;

	yydebug = false;

	if (argc != 2) {
		fprintf(stderr, "\n");
		fprintf(stderr, "Usage: %s {prefix}\n", argv[0]);
		fprintf(stderr, "\n");
		fprintf(stderr, "This will read the file {prefix}.dfs and write {prefix}.v.\n");
		fprintf(stderr, "\n");
		return 1;
	} else
		prefix = argv[1];

	f = fopen((prefix + ".dfs").c_str(), "r");
	if (f == NULL) {
		fprintf(stderr, "Can't open `%s.dfs' for reading: %s\n", prefix.c_str(), strerror(errno));
		return 1;
	}

	yyrestart(f);
	if (yyparse() != 0)
		return 1;
	yylex_destroy();
	fclose(f);

	root->optimize();
	// root->dump();
	// root->dump_code();
	// return 0;

	DfsProg prog;
	root->make_prog(prog);
	prog.optimize();
	// prog.dump();
	// return 0;

	DfsSched *sched = NULL;
	if (prog.set_scheduler.empty() || prog.set_scheduler == "greedy")
		sched = new DfsSchedGreedy(&prog);
	else if (prog.set_scheduler == "minisat")
		sched = new DfsSchedMinisat(&prog);
	else {
		fprintf(stderr, "Invalid scheduler setting: %s\n", prog.set_scheduler.c_str());
		return 1;
	}

	bool ok = sched->schedule();
	// sched->dump();
	delete sched;

	// prog.dump();
	// return 0;

	if (ok)
	{
		f = fopen((prefix + ".map").c_str(), "w");
		if (f == NULL) {
			fprintf(stderr, "Can't open `%s.map' for writing: %s\n", prefix.c_str(), strerror(errno));
			return 1;
		}
		DfsMap mapper(&prog);
		mapper.pack_greedy();
		mapper.map(f);
		fclose(f);

		f = fopen((prefix + ".v").c_str(), "w");
		if (f == NULL) {
			fprintf(stderr, "Can't open `%s.v' for writing: %s\n", prefix.c_str(), strerror(errno));
			return 1;
		}
		DfsMapSimpleVerilog vlog_mapper(&prog);
		vlog_mapper.pack_greedy();
		vlog_mapper.map(f);
		fclose(f);

		printf("Schedule OK.\n");
	}
	else
		printf("Scheduler FAILED.\n");

	delete root;
	root = NULL;

	return !ok;
}

