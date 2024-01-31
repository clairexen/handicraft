%{

#include "gatesynth.h"
#include "parser.tab.h"

#define YY_INPUT(buf,result,max_size) result = yy_input_func(buf, max_size);

const char *yycharp;
int yy_input_func(char *buf, int max_size);

%}

%option yylineno
%option noyywrap
%option nounput

%%

[a-zA-Z][a-zA-Z0-9_]* {
	yylval.string = strdup(yytext);
	return TOK_ID;
}

0 { yylval.value = false; return TOK_VAL; }
1 { yylval.value = true; return TOK_VAL; }

[\r\n\t ] /* ignore whitespaces */

. {
	return *yytext;
}

%%

int yy_input_func(char *buf, int max_size)
{
	int i;

	if (yycharp == NULL) {
		i = fread(buf, 1, max_size, stdin);
		return i > 0 ? i : 0;
	}

	for (i = 0; i < max_size && *yycharp; i++)
		buf[i] = *(yycharp++);
	return i;
}

