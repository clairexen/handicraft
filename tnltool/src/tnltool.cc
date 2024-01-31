/*
 *  tnltool - Transistor Netlist Tool
 *
 *  Copyright (C) 2011 Clifford Wolf <clifford@clifford.at> and
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

#include "tnl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

extern FILE *yyin;
extern int yydebug;
extern int yyparse();

Tnl *tnl = NULL;

int main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "Usage: %s <filename-prefix>\n", argv[0]);
		exit(1);
	}

	std::string prefix = std::string(argv[1]);
	std::string infile = prefix + ".tnl";
	yyin = fopen(infile.c_str(), "r");

	if (yyin == NULL) {
		fprintf(stderr, "Can't open input file `%s': %s\n",
				infile.c_str(), strerror(errno));
		exit(1);
	}

	// yydebug = 1;
	yyparse();

	tnl->check();
	tnl->mergeRedundant();
	tnl->enumerate();
	tnl->bias();

	if (tnl->opt("backend", "csim").compare("csim") == 0)
		tnl->backend_csim(prefix);
	else {
		fprintf(stderr, "Unkown backend `%s'.\n",
				tnl->opt("backend", "csim").c_str());
		exit(1);
	}

	for (auto i = tnl->options.begin(); i != tnl->options.end(); i++) {
		if (tnl->options_used[i->first] || i->second.size() == 0)
			continue;
		fprintf(stderr, "WARNING: Unused option `%s'.\n", i->first.c_str());
	}

	delete tnl;
	tnl = NULL;

	return 0;
}

