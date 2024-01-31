/*
 *  VLNLP - Simple C++ Verilog Netlist Parser/Processor
 *
 *  Copyright (C) 2012  Clifford Wolf <clifford@clifford.at>
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

#ifndef VLNLP_H
#define VLNLP_H

#include <stdio.h>
#include <stdint.h>
#include <string>
#include <vector>
#include <map>

namespace VLNLP {
	struct Value;
	struct Netlist;
	struct Module;
	struct Wire;
	struct Cell;
	struct SigSpec;
}

struct VLNLP::Value
{
	size_t len_in_bits;
	std::vector<uint8_t> data;
	std::string original_str;
	int integer_value;

	Value(const char *str = "");
};

struct VLNLP::Netlist
{
	std::map<std::string, VLNLP::Module*> modules;

	Netlist();
	~Netlist();

	void parse(FILE *f, bool debug = false);
	void fixup();
	void dump(FILE *f);
};

struct VLNLP::Module
{
	std::string name;
	VLNLP::Netlist *parent;

	std::vector<VLNLP::Wire*> ports;
	std::map<std::string, VLNLP::Wire*> wires;
	std::map<std::string, VLNLP::Cell*> cells;
	std::vector<std::pair<VLNLP::SigSpec*, VLNLP::SigSpec*>> assignments;

	Module(VLNLP::Netlist*);
	~Module();

	void dump(FILE *f);
};

struct VLNLP::Wire
{
	std::string name;
	VLNLP::Module *parent;

	int range_left, range_right;
	bool is_input, is_output;
	bool stub_decl;

	Wire(VLNLP::Module*);
	~Wire();

	void dump(FILE *f);
};

struct VLNLP::Cell
{
	std::string name, type_name;
	VLNLP::Module *parent, *type;

	std::map<std::string, VLNLP::Value> parameters;
	std::map<std::string, VLNLP::SigSpec*> connections;

	Cell(VLNLP::Module*);
	~Cell();

	void dump(FILE *f);
};

struct VLNLP::SigSpec
{
	VLNLP::Module *parent;

	struct SigChunk {
		int len, offset;
		Value value;
		Wire *wire;
		void dump(FILE *f);
	};

	int total_len;
	std::vector<SigChunk> chunks;

	SigSpec(VLNLP::Module*);
	~SigSpec();

	void dump(FILE *f);
};

// lexer and parser
extern int vlnlp_yydebug;
extern int vlnlp_yylex();
extern int vlnlp_yyget_lineno();
extern int vlnlp_yylex_destroy();
extern int vlnlp_yyparse();
extern void vlnlp_yyerror(char const *s);
extern void vlnlp_yyrestart(FILE *f);
extern VLNLP::Netlist *vlnlp_parser_netlist;

#endif
