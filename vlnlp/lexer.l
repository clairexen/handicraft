/*
 *  Simple C++ Verilog Netlist Parser/Processor
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
#include "parser.tab.h"
static char *strdup_and_escape(const char *str);
%}

%option yylineno
%option noyywrap
%option prefix="vlnlp_yy"

%x comment

%%

"module"	{ return TOK_MODULE; }
"endmodule"	{ return TOK_ENDMODULE; }

"input"		{ return TOK_INPUT; }
"inout"		{ return TOK_INOUT; }
"output"	{ return TOK_OUTPUT; }
"wire"		{ return TOK_WIRE; }
"assign"	{ return TOK_ASSIGN; }

[0-9]+(\'[bodh][0-9a-fA-FzxZX]+)? {
	vlnlp_yylval.string = strdup(yytext);
	return TOK_CONST;
}

\"(\\.|[^"])*\" {
	vlnlp_yylval.string = strdup(yytext);
	return TOK_CONST;
}

[a-zA-Z_$][0-9a-zA-Z_$]* {
	vlnlp_yylval.string = strdup_and_escape(yytext);
	return TOK_ID;
}

\\[^ \t\r\n]+ {
	vlnlp_yylval.string = strdup_and_escape(yytext+1);
	return TOK_ID;
}

"/*" BEGIN(comment);
<comment>"*/" BEGIN(INITIAL);
<comment>.|\n

"//"[^\n]*\n

[ \t\n\r]       /* ignore whitespaces */
.		{ return *yytext; }

%%

char *strdup_and_escape(const char *str)
{
	for (int i = 0; str[i] != 0; i++) {
		if (i > 0 && '0' <= str[i] && str[i] <= '9')
			continue;
		if ('a' <= str[i] && str[i] <= 'z')
			continue;
		if ('A' <= str[i] && str[i] <= 'Z')
			continue;
		if (str[i] == '_')
			continue;
		int len = strlen(str);
		char *new_str = (char*)malloc(len + 3);
		new_str[0] = '\\';
		memcpy(new_str+1, str, len);
		new_str[len+1] = ' ';
		new_str[len+2] = 0;
		return new_str;
	}
	return strdup(str);
}

// this is a hack to force gcc not to generate the following error msgs:
//   lex.yy.c:1327: warning: ‘yyunput’ defined but not used
void *avoid_gcc_yyunput_input_warnings() {
        return (void*)&yyunput;
}

