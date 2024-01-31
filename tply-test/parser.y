%{

procedure yybuf_sigdata_rst;
begin
	yybuf_sigdata := TSignal.Create;
	yybuf_sigdata.inputName := '';
	yybuf_sigdata.outputName := '';
	yybuf_sigdata.isParameter := false;
end;

procedure yyerror ( msg : String );
begin
	WriteLn('XX ' + msg);
	Halt(1);
end;

%}
 
%token TOK_BEGIN TOK_END
%token TOK_SIGNAL TOK_WORD TOK_UNIT TOK_FSM
%token TOK_PARAMETER TOK_REGISTER
%token TOK_ASSIGN TOK_STORE
%token TOK_IF TOK_GOTO TOK_NEXT

%left TOK_OR
%left TOK_AND
%left TOK_NOT

%token <String> TOK_ID TOK_INPUT TOK_OUTPUT
%token <Integer> TOK_NUM

%type <String> cond
 
%%
 
progtext:
	statement progtext |
	/* nothing */;

statement:
	TOK_SIGNAL { yybuf_sigdata_rst; } sigattr TOK_ID ';' {
		if yysynth.CheckId($4) then
			yyerror('Duplicate declaration of identifier ' + $4 + '.'); 
		yysynth.FSignalList[$4] := yybuf_sigdata;
		// WriteLn('Signal: ', $4); WriteLn;
	} |
	TOK_WORD { yybuf_sigdata_rst; } sigattr TOK_ID ';' {
		if yysynth.CheckId($4) then
			yyerror('Duplicate declaration of identifier ' + $4 + '.'); 
		yysynth.FWordList[$4] := yybuf_sigdata;
		// WriteLn('Word: ', $4); WriteLn;
	} |
	TOK_UNIT TOK_ID TOK_ID {
		if yysynth.CheckId($3) then
			yyerror('Duplicate declaration of identifier ' + $3 + '.'); 
		yybuf_unitdata := TUnitInstance.Create;
		yybuf_unitdata.unitTypeName := $2;
		yybuf_unitdata.signalMap := TSignalMap.Create;
		yysynth.FUnitInstances[$3] := yybuf_unitdata;
	} TOK_BEGIN unitbody TOK_END {
		// WriteLn(Format('Unit: %s (%s)', [$3, $2])); WriteLn;
	} |
	TOK_FSM TOK_ID {
		if yysynth.CheckId($2) then
			yyerror('Duplicate declaration of identifier ' + $2 + '.'); 
		yybuf_fsmdata := TFSM.Create;
		yysynth.FFSMList[$2] := yybuf_fsmdata;
		yybuf_statedata := TFSMStateInsn.Create('LIST');
		yybuf_fsmdata.initState := '';
		yybuf_fsmdata.defaultActions := yybuf_statedata;
		yybuf_fsmdata.inputSignals := TSignalSet.Create;
		yybuf_fsmdata.outputSignals := TSignalSet.Create;
		yybuf_fsmdata.stateList := TFSMStateList.Create;
		yybuf_fsmdata.state2next := TFSMState2Next.Create;
		yybuf_fsmdata.transitions := TTransitionList.Create;
		yycurrentstate := '';
	} TOK_BEGIN fsmbody TOK_END {
		// WriteLn('FSM: ', $2); WriteLn;
	};

sigattr:
	TOK_INPUT sigattr {
		yybuf_sigdata.inputName := $1;
		// WriteLn('INPUT: ', $1);
	} |
	TOK_OUTPUT sigattr {
		yybuf_sigdata.outputName := $1;
		// WriteLn('OUTPUT: ', $1);
	} |
	TOK_PARAMETER sigattr {
		yybuf_sigdata.isParameter := true;
		// WriteLn('PARAMETER');
	} |
	/* nothing */;

unitbody:
	unitbody TOK_ID ':' TOK_ID ';' {
		if not yysynth.CheckId($4) then
			yyerror('References undeclared identifier ' + $4 + '.'); 
		yybuf_unitdata.signalMap[$2] := $4;
		// WriteLn(Format('UNIG SIGNAL ASSIGNMENT: %s <- %s', [$2, $4]));
	} |
	/* nothing */;

fsmbody:
	fsmbody TOK_ID ':' {
		if yybuf_fsmdata.initState = '' then
			yybuf_fsmdata.initState := $2
		else
			yybuf_fsmdata.state2next[yycurrentstate] := $2;
		yycurrentstate := $2;
		yybuf_statedata := TFSMStateInsn.Create('LIST');
		yybuf_statedata.insnType := 'LIST';
		yybuf_fsmdata.stateList[$2] := yybuf_statedata;
		// yybuf_fsmdata.defaultActions.cloneTo(yybuf_statedata);
		// WriteLn('NEW STATE: ', $2);
	} |
	fsmbody TOK_ID TOK_ASSIGN TOK_NUM ';' {
		if yybuf_fsmdata.outputSignals.IndexOf($2) < 0 then
			yybuf_fsmdata.outputSignals.Add($2);
		yybuf_statedata.childNodes.Add(TFSMStateInsn.Create('ASSIGN', $2, Format('%d', [$4])));
		// WriteLn(Format('NEW ASSIGNMENT: %s := %d', [$2, $4]));
	} |
	fsmbody TOK_ID TOK_ASSIGN TOK_NUM TOK_IF cond ';' {
		if yybuf_fsmdata.outputSignals.IndexOf($2) < 0 then
			yybuf_fsmdata.outputSignals.Add($2);
		yyinsstack[yyinsstack_idx] := yybuf_statedata;
		yybuf_statedata := TFSMStateInsn.Create('IF', $6);
		yybuf_statedata.childNodes.Add(TFSMStateInsn.Create('ASSIGN', $2, Format('%d', [$4])));
		yyinsstack[yyinsstack_idx].childNodes.Add(yybuf_statedata);
		yybuf_statedata := yyinsstack[yyinsstack_idx];
		// WriteLn(Format('NEW CONDITIONAL ASSIGNMENT: %s := %d', [$2, $4]));
	} |
	fsmbody TOK_GOTO TOK_ID ';' {
		yybuf_statedata.childNodes.Add(TFSMStateInsn.Create('GOTO', $3));
		// WriteLn('GOTO STATE: ', $3);
	} |
	fsmbody TOK_NEXT ';' {
		yybuf_statedata.childNodes.Add(TFSMStateInsn.Create('NEXT'));
		// WriteLn('GOTO NEXT STATE');
	} |
	fsmbody TOK_GOTO TOK_ID TOK_IF cond ';' {
		yyinsstack[yyinsstack_idx] := yybuf_statedata;
		yybuf_statedata := TFSMStateInsn.Create('IF', $5);
		yybuf_statedata.childNodes.Add(TFSMStateInsn.Create('GOTO', $3));
		yyinsstack[yyinsstack_idx].childNodes.Add(yybuf_statedata);
		yybuf_statedata := yyinsstack[yyinsstack_idx];
		// WriteLn('CONDITIONALLY GOTO STATE: ', $3);
	} |
	fsmbody TOK_NEXT TOK_IF cond ';' {
		yyinsstack[yyinsstack_idx] := yybuf_statedata;
		yybuf_statedata := TFSMStateInsn.Create('IF', $4);
		yybuf_statedata.childNodes.Add(TFSMStateInsn.Create('NEXT'));
		yyinsstack[yyinsstack_idx].childNodes.Add(yybuf_statedata);
		yybuf_statedata := yyinsstack[yyinsstack_idx];
		// WriteLn('CONDITIONALLY GOTO NEXT STATE');
	} |
	fsmbody TOK_IF cond TOK_BEGIN {
		yyinsstack[yyinsstack_idx] := yybuf_statedata;
		yybuf_statedata := TFSMStateInsn.Create('IF', $3);
		yyinsstack_idx := yyinsstack_idx + 1;
		// WriteLn('BEGIN CONDITIONAL BLOCK');
	} fsmbody TOK_END {
		yyinsstack_idx := yyinsstack_idx - 1;
		yyinsstack[yyinsstack_idx].childNodes.Add(yybuf_statedata);
		yybuf_statedata := yyinsstack[yyinsstack_idx];
		// WriteLn('END CONDITIONAL BLOCK');
	} |
	/* nothing */;

cond:
	TOK_ID {
		$$ := $1;
		if yybuf_fsmdata.inputSignals.IndexOf($1) < 0 then
			yybuf_fsmdata.inputSignals.Add($1);
		// WriteLn('IF ', $1);
	} |
	'@' TOK_ID {
		$$ := '@' + $2;
		// WriteLn('IF @', $2);
	} |
	cond TOK_AND cond {
		$$ := '(' + $1 + '&' + $3 + ')';
		// WriteLn('*AND*');
	} |
	cond TOK_OR cond {
		$$ := '(' + $1 + '|' + $3 + ')';
		// WriteLn('*OR*');
	} |
	TOK_NOT cond {
		$$ := '(' + '!' + $2 + ')';
		// WriteLn('*NOT*');
	} |
	'(' {
		// WriteLn('*OPEN-COND*');
	} cond ')' {
		$$ := $3;
		// WriteLn('*CLOSE-COND*');
	};
 
%%
