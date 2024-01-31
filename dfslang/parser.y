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

%{
#include "dfslang.h"
#include <stdio.h>
#include <stdlib.h>
void yyerror (char const *s) {
	fprintf(stderr, "Parser error in line %d: %s\n", yyget_lineno(), s);
	exit(1);
}
%}

%union {
	AstNode *ast;
	int value;
	char *id;
}

%token TOK_INPUT TOK_OUTPUT TOK_SET TOK_FOR TOK_CORE TOK_TASK TOK_BEGIN TOK_END TOK_USE TOK_AS
%token <value> TOK_TYPE TOK_VALUE
%token <id> TOK_ID

%left OP_EQ OP_NE
%left '<' OP_LE OP_GE '>'
%left '+' '-'
%left '*' '/' '%'

%type <ast> statements statement
%type <ast> expressions expression expr
%type <ast> vars var dims ranges range

%expect 0
%debug

%%

input:
	statements {
		if (root)
			delete root;
		root = $1;
	};

statements:
	statements statement {
		$$ = $1;
		$$->children.push_back($2);
	} |
	/* empty */ {
		$$ = new AstNode("stmts");
	};

statement:
	TOK_SET TOK_ID expression ';' {
		$$ = new AstNode(std::string("set_") + $2, $3);
	} |
	TOK_INPUT TOK_TYPE vars ';' {
		$$ = new AstNode("input", new AstNode("type", $2), $3);
	} |
	TOK_OUTPUT TOK_TYPE vars ';' {
		$$ = new AstNode("output", new AstNode("type", $2), $3);
	} |
	TOK_TYPE vars ';' {
		$$ = new AstNode("variable", new AstNode("type", $1), $2);
	} |
	TOK_USE TOK_ID TOK_AS TOK_ID ';' {
		$$ = new AstNode("use",
				new AstNode("mod", $2),
				new AstNode("var", $4));
		free($2);
		free($4);
	} |
	TOK_FOR ranges TOK_BEGIN statements TOK_END {
		$$ = new AstNode("for", $2, $4);
	} |
	TOK_CORE TOK_ID expression TOK_BEGIN statements TOK_END {
		$$ = new AstNode("core", $2, $3, $5);
		free($2);
	} |
	TOK_TASK TOK_ID '(' vars ')' TOK_BEGIN statements TOK_END {
		$$ = new AstNode("task", $2, $4, $7);
		free($2);
	} |
	TOK_ID '(' expressions ')' ';' {
		$$ = new AstNode("call", $1, $3);
		free($1);
	} |
	expression '=' expression ';' {
		$$ = new AstNode("assign", $1, $3);
	};

vars:
	var {
		$$ = $1;
	} |
	var ',' vars {
		$$ = $1;
		$$->children.push_back($3);
	};

var:
	TOK_ID dims {
		$$ = new AstNode("var", $1, $2);
		free($1);
	};

expressions:
	expr {
		$$ = $1;
	} |
	expr ',' expressions {
		$$ = $1;
		$$->children.push_back($3);
	};

dims:
	'[' expression ']' dims {
		$$ = new AstNode("dim", $2, $4);
	} |
	/* empty */ {
		$$ = NULL;
	};

ranges:
	ranges ',' range {
		$$ = $1;
		$$->children.push_back($3);
	} |
	range {
		$$ = $1;
	};

range:
	TOK_ID '=' expression ':' expression {
		$$ = new AstNode("range",
				new AstNode("var", $1),
				$3, $5);
		free($1);
	};

expr:
	expression {
		$$ = new AstNode("expr", $1);
	};

expression:
	TOK_ID '.' TOK_ID dims {
		$$ = new AstNode("ref",
				new AstNode("var", $1),
				new AstNode("port", $3),
				$4);
		free($1);
		free($3);
	} |
	TOK_ID dims {
		$$ = new AstNode("ref",
				new AstNode("var", $1), $2);
		free($1);
	} |
	'(' expression ')'		{ $$ = $2; } |
	TOK_VALUE 			{ $$ = new AstNode("value",  $1); } |
	'-' expression			{ $$ = new AstNode("neg", $2); } |
	'+' expression			{ $$ = $2; } |
	expression '-' expression	{ $$ = new AstNode("op(-)",  $1, $3); } |
	expression '+' expression	{ $$ = new AstNode("op(+)",  $1, $3); } |
	expression '*' expression	{ $$ = new AstNode("op(*)",  $1, $3); } |
	expression '/' expression	{ $$ = new AstNode("op(/)",  $1, $3); } |
	expression '%' expression	{ $$ = new AstNode("op(%)",  $1, $3); } |
	expression '<' expression	{ $$ = new AstNode("op(<)",  $1, $3); } |
	expression OP_LE expression	{ $$ = new AstNode("op(<=)", $1, $3); } |
	expression OP_EQ expression	{ $$ = new AstNode("op(==)", $1, $3); } |
	expression OP_NE expression	{ $$ = new AstNode("op(!=)", $1, $3); } |
	expression OP_GE expression	{ $$ = new AstNode("op(>=)", $1, $3); } |
	expression '>' expression	{ $$ = new AstNode("op(>)",  $1, $3); };

%%

