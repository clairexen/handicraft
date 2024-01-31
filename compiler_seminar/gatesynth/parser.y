
%{
#include "gatesynth.h"
struct workspace *yyworkspace;
%}

%union {
	char *string;
	bool value;
	struct aitree *aitree;
}

%type <aitree> sum product atom
%token <string> TOK_ID
%token <value> TOK_VAL

%debug

%%

input:
	assign_list |
	sum {
		yyworkspace->tree = $1;
	};

assign_list:
	/* empty */ |
	TOK_ID '=' sum ';' assign_list {
		if (yyworkspace->forest.count($1) > 0)
			yyerror("Identifier collision");
		yyworkspace->forest[$1] = $3;
		free($1);
	};

sum:
	product {
		$$ = $1;
	} |
	sum '+' product {
		$$ = new aitree($1, $3);
		$$->parents[0]->invert();
		$$->parents[1]->invert();
		$$->invert();
	};

product:
	atom {
		$$ = $1;
	} |
	product '*' atom {
		$$ = new aitree($1, $3);
	};

atom:
	'(' sum ')' {
		$$ = $2;
	} |
	TOK_ID {
		$$ = new aitree($1);
		yyworkspace->inputs.insert($1);
		free($1);
	} |
	TOK_VAL {
		$$ = new aitree;
		if ($1 == false)
			$$->invert();
	} |
	atom '\'' {
		$$ = $1;
		$$->invert();
	};

%%

void yyerror(const char *msg)
{
	printf("Fatal parser/lexer error in line %d: %s\n", yyget_lineno(), msg);
	exit(1);
}

