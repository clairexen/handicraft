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

%{

#include "tnl.h"
#include <list>
#include <stdio.h>
#include <stdarg.h>

extern int yydebug;
extern int yylineno;
extern int yylex(void);
void yyerror(char const *s);
void yyerrorf(char const *fmt, ...);

extern Tnl *tnl;
std::list<std::string> id_list;
std::list<std::string> id_list2;

%}

%union {
	std::string *str;
}

%token STATES
%token COMBINE
%token ASCEND
%token STATELABELS
%token SWITCH
%token OPTION
%token ON
%token OFF
%token END

%token ID
%token EOL

%type <str> ID

%expect 1
%debug

%%

input:
	{
		tnl = new Tnl;
	}
	statements
	{
		id_list.clear();
		id_list2.clear();
		fprintf(stderr, "TNL input file parsed.\n");
	};

statements:
	/* empty */ |
	statement statements;

eol:
	EOL | EOL eol;

statement:
	eol |
	STATES id_list EOL {
		for (auto i = id_list.begin(); i != id_list.end(); i++) {
			if (tnl->str2state.count(*i) != 0)
				yyerrorf("Duplicate definition of state `%s'", i->c_str());
			tnl->str2state[*i] = tnl->numStates;
			tnl->state2str[tnl->numStates] = *i;
			for (int j = 0; j < tnl->numStates; j++) {
				tnl->stateCombine[std::pair<int, int>(tnl->numStates, j)] = -1;
				tnl->stateCombine[std::pair<int, int>(j, tnl->numStates)] = -1;
			}
			tnl->stateCombine[std::pair<int, int>(tnl->numStates, tnl->numStates)] = -1;
			tnl->stateAscend[tnl->numStates] = -1;
			tnl->numStates++;
		}
	} |
	COMBINE id_list ':' {
		id_list2 = id_list;
	} EOL combine_body END EOL |
	ASCEND EOL ascend_body END EOL |
	SWITCH ID ':' EOL ON id_list EOL { id_list2 = id_list; }
			OFF id_list EOL END EOL {
		if (tnl->switchTypes.count(*$2) != 0)
			yyerrorf("Duplicate definition of switch type `%s'", $2->c_str());
		Tnl::SwitchType *st = new Tnl::SwitchType;
		tnl->switchTypes[*$2] = st;
		st->id = *$2;
		for (auto i = id_list2.begin(); i != id_list2.end(); i++) {
			if (tnl->str2state.count(*i) == 0)
				yyerrorf("Unknown state `%s'", i->c_str());
			st->onStates.insert(tnl->str2state[*i]);
		}
		for (auto i = id_list.begin(); i != id_list.end(); i++) {
			if (tnl->str2state.count(*i) == 0)
				yyerrorf("Unknown state `%s'", i->c_str());
			st->offStates.insert(tnl->str2state[*i]);
		}
		delete $2;
	} |
	OPTION ID ':' id_list {
		tnl->options[*$2].clear();
		for (auto i = id_list.begin(); i != id_list.end(); i++)
			tnl->options[*$2].push_back(*i);
		delete $2;
	} |
	'N' id_list ':' { id_list2 = id_list; } id_list EOL {
		Tnl::Net *n = new Tnl::Net;
		for (auto i = id_list2.begin(); i != id_list2.end(); i++) {
			if (tnl->id2net.count(*i) != 0)
				yyerrorf("Duplicate definition of net `%s'", i->c_str());
			tnl->id2net[*i] = n;
			n->ids.insert(*i);
		}
		for (auto i = id_list.begin(); i != id_list.end(); i++) {
			if (!i->compare("supply")) {
				n->supply = true;
				continue;
			}
			if (tnl->str2state.count(*i) == 0)
				yyerrorf("Unknown state `%s'", i->c_str());
			n->driveState = tnl->str2state[*i];
		}
		tnl->nets.insert(n);
	} |
	'T' id_list ':' { id_list2 = id_list; } id_list EOL {
		Tnl::Switch *s = new Tnl::Switch;
		for (auto i = id_list2.begin(); i != id_list2.end(); i++) {
			if (tnl->id2switch.count(*i) != 0)
				yyerrorf("Duplicate definition of transistor `%s'", i->c_str());
			tnl->id2switch[*i] = s;
			s->ids.insert(*i);
		}
		if (id_list.size() != 4)
			yyerror("Invalid number of arguments for transistor definition");
		auto i = id_list.begin();
		std::string type = *(i++);
		std::string gate = *(i++);
		std::string c1 = *(i++);
		std::string c2 = *(i++);
		if (tnl->switchTypes.count(type) == 0)
			yyerrorf("Unknown transistor type `%s'", type.c_str());
		if (tnl->id2net.count(gate) == 0)
			yyerrorf("Unknown net `%s'", gate.c_str());
		if (tnl->id2net.count(c1) == 0)
			yyerrorf("Unknown net `%s'", c1.c_str());
		if (tnl->id2net.count(c2) == 0)
			yyerrorf("Unknown net `%s'", c2.c_str());
		s->type = tnl->switchTypes[type];
		s->gate = tnl->id2net[gate];
		s->cc[0] = tnl->id2net[c1];
		s->cc[1] = tnl->id2net[c2];
		tnl->id2net[gate]->switchGates.insert(s);
		tnl->id2net[c1]->switchCC.insert(s);
		tnl->id2net[c2]->switchCC.insert(s);
		tnl->switches.insert(s);
	};

id_list:
	{ id_list.clear() } id_list_tail;

id_list_tail:
	ID { id_list.push_back(*$1); delete $1; } |
	ID { id_list.push_back(*$1); delete $1; } id_list_tail;

combine_body:
	combine_line | combine_body combine_line;

combine_line:
	ID ':' id_list EOL {
		if (tnl->str2state.count(*$1) == 0)
			yyerrorf("Unknown state `%s'", $1->c_str());
		if (id_list.size() != id_list2.size())
			yyerrorf("Length of line does not match header line (%d != %d)",
					id_list.size(), id_list2.size());
		
		auto i = id_list.begin();
		auto j = id_list2.begin();
		while (i != id_list.end() && j != id_list2.end()) {
			if (tnl->str2state.count(*i) == 0)
				yyerrorf("Unknown state `%s'", i->c_str());
			if (tnl->str2state.count(*j) == 0)
				yyerrorf("Unknown state `%s'", j->c_str());
			int rowState = tnl->str2state[*$1];
			int colState = tnl->str2state[*j];
			int state = tnl->str2state[*i];
			tnl->stateCombine[std::pair<int, int>(rowState, colState)] = state;
			i++, j++;
		}
		delete $1;
	};

ascend_body:
	ascend_line | ascend_body ascend_line;

ascend_line:
	ID ':' ID EOL {
		if (tnl->str2state.count(*$1) == 0)
			yyerrorf("Unknown state `%s'", $1->c_str());
		if (tnl->str2state.count(*$3) == 0)
			yyerrorf("Unknown state `%s'", $3->c_str());
		int state1 = tnl->str2state[*$1];
		int state2 = tnl->str2state[*$3];
		tnl->stateAscend[state1] = state2;
		delete $1;
		delete $3;
	};

%%

void yyerror(char const *s)
{
	fprintf(stderr, "Parser error in line %d: %s\n", yylineno, s);
	exit(1);
}

void yyerrorf(char const *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "Parser error in line %d:", yylineno);
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, "\n");
	va_end(ap);
	exit(1);
}

std::string Tnl::opt(std::string key, std::string def) {
	if (options.count(key) == 0 || options[key].size() == 0)
		return def;
	if (options[key].size() != 1) {
		fprintf(stderr, "Option `%s' only accepts one parameter.\n", key.c_str());
		exit(1);
	}
	options_used[key] = true;
	return options[key].front();
}

bool Tnl::checkopt(std::string key) {
	if (options.count(key) == 0 || options[key].size() == 0)
		return false;
	options_used[key] = true;
	return true;
}

