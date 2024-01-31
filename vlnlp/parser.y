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

%{

#include "vlnlp.h"
#include <stdlib.h>

using namespace VLNLP;

void vlnlp_yyerror(char const *s) {
        fprintf(stderr, "Parser error in line %d: %s\n", vlnlp_yyget_lineno(), s);
	// abort();
        exit(1);
}

Netlist *vlnlp_parser_netlist;
Module *vlnlp_parser_module;
Cell *vlnlp_parser_cell;

int vlnlp_parser_wire_type;
int vlnlp_parser_range_left;
int vlnlp_parser_range_right;

%}

%name-prefix "vlnlp_yy"

%union {
        char *string;
	int number;
	struct { int left, right; } range;
	VLNLP::SigSpec *sigspec;
}

%token <string> TOK_CONST TOK_ID
%token TOK_MODULE TOK_ENDMODULE TOK_INPUT TOK_INOUT TOK_OUTPUT TOK_WIRE TOK_ASSIGN

%type <sigspec> optional_signal_spec signal_spec signal_spec_list

%expect 0
%debug

%%

input: module input | /* empty */;

module:
	TOK_MODULE TOK_ID {
		if (vlnlp_parser_netlist->modules.count($2) != 0)
			vlnlp_yyerror("Scope error");
		vlnlp_parser_module = new Module(vlnlp_parser_netlist);
		vlnlp_parser_module->name = $2;
		vlnlp_parser_netlist->modules[$2] = vlnlp_parser_module;
		free($2);
	} '(' module_args ')' ';' module_body TOK_ENDMODULE;

module_args:
	/* empty */ |
	module_args_list;

module_args_list:
	module_single_arg | module_args_list ',' module_single_arg;

module_single_arg:
	TOK_ID {
		if (vlnlp_parser_module->wires.count($1) != 0)
			vlnlp_yyerror("Scope error");
		Wire *w = new Wire(vlnlp_parser_module);
		w->name = $1;
		w->stub_decl = true;
		vlnlp_parser_module->wires[$1] = w;
		vlnlp_parser_module->ports.push_back(w);
		free($1);
	};

module_body:
	/* empty */ |
	module_body wire_statement |
	module_body cell_statement |
	module_body assign_statement;

wire_statement:
	wire_type wire_range wire_names ';';

wire_type:
	TOK_INPUT  { vlnlp_parser_wire_type = 1; } |
	TOK_INOUT  { vlnlp_parser_wire_type = 3; } |
	TOK_OUTPUT { vlnlp_parser_wire_type = 2; } |
	TOK_WIRE   { vlnlp_parser_wire_type = 0; };

wire_range:
	/* empty */ {
		vlnlp_parser_range_left = 0;
		vlnlp_parser_range_right = 0;
	} |
	'[' TOK_CONST ':' TOK_CONST ']' {
		vlnlp_parser_range_left = Value($2).integer_value;
		vlnlp_parser_range_right = Value($4).integer_value;
		free($2);
		free($4);
	};

wire_names:
	wire_single_name | wire_names ',' wire_single_name;

wire_single_name:
	TOK_ID {
		Wire *w = NULL;
		if (vlnlp_parser_wire_type != 0) {
			if (vlnlp_parser_module->wires.count($1) == 0)
				vlnlp_yyerror("Scope error");
			w = vlnlp_parser_module->wires[$1];
			if (!w->stub_decl)
				vlnlp_yyerror("Scope error");
			w->stub_decl = false;
		} else {
			if (vlnlp_parser_module->wires.count($1) != 0)
				vlnlp_yyerror("Scope error");
			w = new Wire(vlnlp_parser_module);
			w->name = $1;
			vlnlp_parser_module->wires[$1] = w;
		}
		w->range_left = vlnlp_parser_range_left;
		w->range_right = vlnlp_parser_range_right;
		w->is_input = (vlnlp_parser_wire_type & 1) != 0;
		w->is_output = (vlnlp_parser_wire_type & 2) != 0;
		free($1);
	};

cell_statement:
	TOK_ID {
		vlnlp_parser_cell = new Cell(vlnlp_parser_module);
		vlnlp_parser_cell->type_name = $1;
		free($1);
	} cell_parameters TOK_ID {
		if (vlnlp_parser_module->cells.count($4) != 0)
			vlnlp_yyerror("Scope error");
		vlnlp_parser_module->cells[$4] = vlnlp_parser_cell;
		vlnlp_parser_cell->name = $4;
		free($4);
	} '(' cell_connections ')' ';';

cell_parameters:
	/* empty */ | '#' '(' ')' |
	'#' '(' cell_parameter_list ')';

cell_parameter_list:
	cell_single_parameter | cell_parameter_list ',' cell_single_parameter;

cell_single_parameter:
	'.' TOK_ID '(' TOK_CONST ')' {
		vlnlp_parser_cell->parameters[$2] = Value($4);
		free($2);
		free($4);
	};

cell_connections:
	cell_single_connection | cell_connections ',' cell_single_connection;

cell_single_connection:
	'.' TOK_ID '(' optional_signal_spec ')' {
		vlnlp_parser_cell->connections[$2] = $4;
		free($2);
	} |
	signal_spec {
		vlnlp_yyerror("Positional arguments are not supportet at the moment.");
		delete $1;
	};

assign_statement:
	TOK_ASSIGN assignment_list ';';

assignment_list:
	single_assignment | assignment_list ',' single_assignment;

single_assignment:
	signal_spec '=' signal_spec {
		vlnlp_parser_module->assignments.push_back(std::pair<VLNLP::SigSpec*, VLNLP::SigSpec*>($1, $3));
	};

optional_signal_spec:
	/* empty */ {
		$$ = new SigSpec(vlnlp_parser_module);
	} |
	signal_spec {
		$$ = $1;
	};
	
signal_spec:
	TOK_ID {
		if (vlnlp_parser_module->wires.count($1) == 0)
			vlnlp_yyerror("Scope error");
		Wire *w = vlnlp_parser_module->wires[$1];
		if (w->stub_decl)
			vlnlp_yyerror("Scope error");
		$$ = new SigSpec(vlnlp_parser_module);
		$$->chunks.push_back(SigSpec::SigChunk());
		SigSpec::SigChunk &chunk = $$->chunks.back();
		chunk.len = w->range_left - w->range_right + 1;
		chunk.offset = w->range_right;
		chunk.wire = w;
		$$->total_len = chunk.len;
		free($1);
	} |
	TOK_ID '[' TOK_CONST ']' {
		if (vlnlp_parser_module->wires.count($1) == 0)
			vlnlp_yyerror("Scope error");
		Wire *w = vlnlp_parser_module->wires[$1];
		if (w->stub_decl)
			vlnlp_yyerror("Scope error");
		$$ = new SigSpec(vlnlp_parser_module);
		$$->chunks.push_back(SigSpec::SigChunk());
		SigSpec::SigChunk &chunk = $$->chunks.back();
		chunk.len = 1;
		chunk.offset = Value($3).integer_value;
		chunk.wire = w;
		if (w->range_left < chunk.offset || chunk.offset < w->range_right)
			vlnlp_yyerror("Range error");
		$$->total_len = chunk.len;
		free($1);
		free($3);
	} |
	TOK_ID '[' TOK_CONST ':' TOK_CONST ']' {
		if (vlnlp_parser_module->wires.count($1) == 0)
			vlnlp_yyerror("Scope error");
		Wire *w = vlnlp_parser_module->wires[$1];
		if (w->stub_decl)
			vlnlp_yyerror("Scope error");
		$$ = new SigSpec(vlnlp_parser_module);
		$$->chunks.push_back(SigSpec::SigChunk());
		SigSpec::SigChunk &chunk = $$->chunks.back();
		chunk.len = Value($3).integer_value - Value($5).integer_value + 1;
		chunk.offset = Value($5).integer_value;
		chunk.wire = w;
		if (w->range_left < chunk.offset || chunk.offset < w->range_right || chunk.len < 1)
			vlnlp_yyerror("Range error");
		if (w->range_left < chunk.offset+chunk.len-1 || chunk.offset+chunk.len-1 < w->range_right)
			vlnlp_yyerror("Range error");
		$$->total_len = chunk.len;
		free($1);
		free($3);
		free($5);
	} |
	TOK_CONST {
		$$ = new SigSpec(vlnlp_parser_module);
		$$->chunks.push_back(SigSpec::SigChunk());
		SigSpec::SigChunk &chunk = $$->chunks.back();
		chunk.wire = NULL;
		chunk.value = Value($1);
		chunk.len = chunk.value.len_in_bits;
		chunk.offset = 0;
		$$->total_len = chunk.len;
		free($1);
	} |
	'{' signal_spec_list '}' {
		$$ = $2;
	};

signal_spec_list:
	signal_spec {
		$$ = $1;
	} |
	signal_spec ',' signal_spec_list {
		for (size_t i = 0; i < $1->chunks.size(); i++) {
			$3->chunks.push_back($1->chunks[i]);
			$3->total_len += $1->chunks[i].len;
		}
		delete $1;
		$$ = $3;
	};

%%

