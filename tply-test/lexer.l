%{

procedure lexer_xtract_inoutname(var dout : String; var din : String);
	var i : Integer;
	var active : Boolean;
begin
	dout := '';
	active := false;
	for i := 1 to length(din) do begin
		if din[i] = ')' then
			active := false;
		if active then
			dout := dout + din[i];
		if din[i] = '(' then
			active := true;
	end
end;

%}

%%

"begin"			return(TOK_BEGIN);
"end"			return(TOK_END);

"signal"		return(TOK_SIGNAL);
"word"			return(TOK_WORD);
"unit"			return(TOK_UNIT);
"fsm"			return(TOK_FSM);

"input("[^)]*")"	begin lexer_xtract_inoutname(yylval.yyString, yytext); return(TOK_INPUT); end;
"output("[^)]*")"	begin lexer_xtract_inoutname(yylval.yyString, yytext); return(TOK_OUTPUT); end;
"parameter"		return(TOK_PARAMETER);
"register"		return(TOK_REGISTER);

":="			return(TOK_ASSIGN);
"<="			return(TOK_STORE);

"if"			return(TOK_IF);
"not"			return(TOK_NOT);
"and"			return(TOK_AND);
"or"			return(TOK_OR);

"goto"			return(TOK_GOTO);
"next"			return(TOK_NEXT);
 
[A-Z][A-Za-z0-9_]*	begin yylval.yyString := yytext; return(TOK_ID); end;
[A-Z][A-Za-z0-9_]*	begin yylval.yyString := yytext; return(TOK_ID); end;
[0-9]+			begin yylval.yyInteger := StrToInt(yytext); return(TOK_NUM); end;
 
[ \t\r\n]		;
"//"[^\n]*		;
[:();@]			return(Integer(yytext[1]));
.			yyerror(Format('Lexer Error: %s', [yytext]));
 
%%
