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
#include "tnl_parser.tab.h"

#include <stack>
#include <stdio.h>

%}

%option yylineno
%option noyywrap
%x include

%%

"STATES"        { return STATES; }
"COMBINE"       { return COMBINE; }
"ASCEND"        { return ASCEND; }
"SWITCH"        { return SWITCH; }
"ON"            { return ON; }
"OFF"           { return OFF; }
"OPTION"        { return OPTION; }
"END"           { return END; }

[_a-z][^ \t\r\n:,;]+ {
	yylval.str = new std::string(yytext);
	return ID;
}

\"[^"]*\" {
	yylval.str = new std::string(yytext);
	yylval.str->erase(0, 1);
	yylval.str->erase(yylval.str->size()-1, 1);
	return ID;
}

[ \t]
"#"[^\r\n]*
[\r\n;]+ { return EOL; }
.        { return *yytext; }

"INCLUDE"[ \t]+[^ \t\n\r]+ {
	char *buf = strdup(yytext);
	char *fn = strtok(buf, " \t");
	fn = strtok(NULL, " \t\n\r");
	yyin = fopen(fn, "r");
	if (yyin == NULL) {
		fprintf(stderr, "Can't open include file `%s'!\n", fn);
		exit(1);
	}
	yypush_buffer_state(yy_create_buffer(yyin, YY_BUF_SIZE));
	free(buf);
}

<<EOF>> {
	fclose(yyin);
	yypop_buffer_state();
	if (!YY_CURRENT_BUFFER) {
		// there is no 'official' flex api to free the buffer stack
		// but we know that it is enough just to free the 'yy_buffer_stack' variable..
		free(yy_buffer_stack);
		yyterminate();
	}
}


%%

// create a non-static wrapper for yyunput() to suppress compiler
// warnings about yyunput() beeing defined but not used..
void yyunput_ext(int a, char *b) { yyunput(a, b); }

