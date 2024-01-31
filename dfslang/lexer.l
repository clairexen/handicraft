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
#include "parser.hh"

#define YY_INPUT(buf,result,max_size) {				\
	int c = getc(yyin);					\
	while (c == '\\') {					\
		c = getc(yyin);				\
		if (c == '\n')					\
			c = getc(yyin);			\
		else						\
			break;					\
	}							\
	if (yydebug)						\
		fprintf(stderr, "Lexer: `%c' (%d)\n",		\
			c >= 32 && c < 128 ? c : '?', c);	\
	result = (c == EOF) ? YY_NULL : (buf[0] = c, 1);	\
}

#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunneeded-internal-declaration"

%}

%option yylineno
%option noyywrap

%x comment

%%

[us][0-9]+ {
	yylval.value = atoi(yytext+1);
	if (yytext[0] == 's')
		yylval.value *= -1;
	return TOK_TYPE;
}

"input"		{ return TOK_INPUT; }
"output"	{ return TOK_OUTPUT; }
"set" 		{ return TOK_SET; }
"for" 		{ return TOK_FOR; }
"core" 		{ return TOK_CORE; }
"task" 		{ return TOK_TASK; }
"begin" 	{ return TOK_BEGIN; }
"end" 		{ return TOK_END; }
"use" 		{ return TOK_USE; }
"as" 		{ return TOK_AS; }

[a-zA-Z_][a-zA-Z0-9_]* {
	yylval.id = strdup(yytext);
	return TOK_ID;
}

[0-9]+ {
	yylval.value = atoi(yytext);
	return TOK_VALUE;
}

"/*" { BEGIN(comment); }
<comment>.    /* ignore comment body */
<comment>\n   /* ignore comment body */
<comment>"*/" { BEGIN(0); }

"//"[^\n]*
[\n\r\t ]

"=="	{ return OP_EQ; }
"!="	{ return OP_NE; }
"<="	{ return OP_LE; }
">="	{ return OP_GE; }

. { return *yytext; }

%%

