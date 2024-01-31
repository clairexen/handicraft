program synth;
{$mode objfpc}{$H+}

uses SysUtils, FGL, yacclib, lexlib;

type
  TSignal = class
    inputName : String;
    outputName : String;
    isParameter : Boolean;
  end;

  TSignalMap = specialize TFPGMap<String, String>;

  TUnitInstance = class
    unitTypeName : String;
    signalMap : TSignalMap;
  end;

  TFSMStateInsn = class;
  TFSMStateInsnList = specialize TFPGList<TFSMStateInsn>;

  TFSMStateInsn = class
    childNodes : TFSMStateInsnList;
    insnType : String;
    arg1, arg2, arg3 : String;
    constructor Create(setType : String);
    constructor Create(setType : String; a1 : String);
    constructor Create(setType : String; a1, a2 : String);
    constructor Create(setType : String; a1, a2, a3 : String);
    procedure CloneTo(target : TFSMStateInsn);
    procedure Dump(indent : String);
  end;

  TFSMStateList = specialize TFPGMap<String, TFSMStateInsn>;
  TFSMState2Next = specialize TFPGMap<String, String>;
  TTransitionList = specialize TFPGMap<String, String>;
  TSignalSet = specialize TFPGList<String>;

  TFSM = class
    initState : String;
    defaultActions : TFSMStateInsn;
    inputSignals : TSignalSet;
    outputSignals : TSignalSet;
    stateList : TFSMStateList;
    state2next : TFSMState2Next;
    transitions : TTransitionList;
    function EvalCond(inputPattern, state, cond : String) : String;
    procedure EvalInsn(var outputPattern, nextState : String; inputPattern, state : String; insn : TFSMStateInsn);
    procedure GenerateTransitions;
    procedure CompressTransitions;
    procedure CollectTransitions;
  end;

  TSignalList = specialize TFPGMap<String, TSignal>;
  TUnitInstanceList = specialize TFPGMap<String, TUnitInstance>;
  TFSMList = specialize TFPGMap<String, TFSM>;

  TSynth = class
    FSignalList : TSignalList;
    FWordList : TSignalList;
    FUnitInstances : TUnitInstanceList;
    FFSMList : TFSMList;
    constructor Create(filename : String);
    function CheckId(id : String) : Boolean;
    procedure Worker_SubstNext;
    procedure Dump;
  end;

var yybuf_sigdata : TSignal;
var yybuf_unitdata : TUnitInstance;
var yybuf_fsmdata : TFSM;
var yybuf_statedata : TFSMStateInsn;
var yyinsstack : array [0..15] of TFSMStateInsn;
var yyinsstack_idx : Integer;
var yycurrentstate : String;
var yysynth : TSynth;

{$H-}
{$include parser.pas}
{$include lexer.pas}
{$H+}

constructor TFSMStateInsn.Create(setType : String);
begin
  childNodes := TFSMStateInsnList.Create;
  insnType := setType;
  arg1 := '';
  arg2 := '';
  arg3 := '';
end;

constructor TFSMStateInsn.Create(setType : String; a1 : String);
begin
  childNodes := TFSMStateInsnList.Create;
  insnType := setType;
  arg1 := a1;
  arg2 := '';
  arg3 := '';
end;

constructor TFSMStateInsn.Create(setType : String; a1, a2 : String);
begin
  childNodes := TFSMStateInsnList.Create;
  insnType := setType;
  arg1 := a1;
  arg2 := a2;
  arg3 := '';
end;

constructor TFSMStateInsn.Create(setType : String; a1, a2, a3 : String);
begin
  childNodes := TFSMStateInsnList.Create;
  insnType := setType;
  arg1 := a1;
  arg2 := a2;
  arg3 := a3;
end;

procedure TFSMStateInsn.CloneTo(target : TFSMStateInsn);
  var i : Integer;
  var minime : TFSMStateInsn;
begin
  if insnType = 'LIST' then begin
    for i := 0 to childNodes.Count-1 do
      childNodes[i].CloneTo(target);
  end else begin
    minime := TFSMStateInsn.Create(insnType, arg1, arg2, arg3);
    for i := 0 to childNodes.Count-1 do
      childNodes[i].CloneTo(minime);
    target.childNodes.Add(minime);
  end;
end;

procedure TFSMStateInsn.Dump(indent : String);
  var i : Integer;
  label write_children;
begin
  if insnType = 'LIST' then begin
    goto write_children;
  end;
  if insnType = 'ASSIGN' then begin
    WriteLn(indent, arg1, ' := ', arg2, ';');
    goto write_children;
  end;
  if insnType = 'IF' then begin
    Write(indent, 'if ');
    for i := 1 to length(arg1) do
      if arg1[i] = '&' then
        Write(' and ')
      else if arg1[i] = '|' then
        Write(' or ')
      else if arg1[i] = '!' then
        Write('not ')
      else
        Write(arg1[i]);
    WriteLn(' begin');
    for i := 0 to childNodes.Count-1 do
      childNodes[i].Dump(indent + '  ');
    WriteLn(indent, 'end');
    exit;
  end;
  if insnType = 'GOTO' then begin
    WriteLn(indent, 'goto ', arg1, ';');
    goto write_children;
  end;
  if insnType = 'NEXT' then begin
    WriteLn(indent, 'next;');
    goto write_children;
  end;
  WriteLn(indent, Format('** %s (%s) (%s) (%s)', [insnType, arg1, arg2, arg3]));
write_children:
  for i := 0 to childNodes.Count-1 do
    childNodes[i].Dump(indent + '  ');
end;

function TFSM.EvalCond(inputPattern, state, cond : String) : String;
  var t1, t2 : String;
  var i, j : Integer;
begin
  EvalCond := 'x';

  if cond = '' then
    exit;

  if cond[1] = '(' then begin
    t1 := '';
    t2 := '';
    j := 1;
    for i := 2 to length(cond) do begin
      if j > 0 then begin
        if cond[i] = '(' then
          j := j + 1;
        if cond[i] = ')' then
          j := j - 1;
        if j > 0 then
          t1 := t1 + cond[i];
      end else
        t2 := t2 + cond[i];
    end;
    cond := EvalCond(inputPattern, state, t1) + t2;
    EvalCond := EvalCond(inputPattern, state, cond);
    exit;
  end;
  
  if cond[1] = '!' then begin
    t1 := '';
    for i := 2 to length(cond) do
      t1 := t1 + cond[i];
    t1 := EvalCond(inputPattern, state, t1);
    if t1 = '1' then
      EvalCond := '0'
    else if t1 = '0' then
      EvalCond := '1';
    exit;
  end;
  
  if cond[1] = '1' then begin
    if cond = '1' then
      EvalCond := cond
    else if cond[2] = '|' then
      EvalCond := '1'
    else if cond[2] = '&' then begin
      t1 := '';
      for i := 3 to length(cond) do
        t1 := t1 + cond[i];
      EvalCond := EvalCond(inputPattern, state, t1);
    end;
    exit;
  end;
  
  if cond[1] = '0' then begin
    if cond = '0' then
      EvalCond := cond
    else if cond[2] = '&' then
      EvalCond := '0'
    else if cond[2] = '|' then begin
      t1 := '';
      for i := 3 to length(cond) do
        t1 := t1 + cond[i];
      EvalCond := EvalCond(inputPattern, state, t1);
    end;
    exit;
  end;

  t1 := '';
  t2 := '';
  for i := 1 to length(cond) do begin
    if (cond[i] <> '(') and (cond[i] <> '!') and (cond[i] <> '&') and (cond[i] <> '|') then
      t1 := t1 + cond[i]
    else begin
      for j := i to length(cond) do
        t2 := t2 + cond[j];
      t1 := EvalCond(inputPattern, state, t1) + t2;
      EvalCond := EvalCond(inputPattern, state, t1);
      exit;
    end;
  end;

  if (length(t1) > 0) and (t1[1] = '@') then
    if t1 = ('@' + state) then
      EvalCond := '1'
    else
      EvalCond := '0'
  else
    EvalCond := inputPattern[1 + inputSignals.IndexOf(t1)];
end;

procedure TFSM.EvalInsn(var outputPattern, nextState : String; inputPattern, state : String; insn : TFSMStateInsn);
  var i : Integer;
begin
  nextState := '';
  if insn.insnType = 'ASSIGN' then
    outputPattern[1 + outputSignals.IndexOf(insn.arg1)] := insn.arg2[1];
  if insn.insnType = 'IF' then begin
    // WriteLn('*** ', state, ':', inputPattern, ' (', insn.arg1, ') -> ', EvalCond(inputPattern, insn.arg1));
    if EvalCond(inputPattern, state, insn.arg1) = '0' then
      exit;
  end;
  if insn.insnType = 'GOTO' then begin
    nextState := insn.arg1;
    exit;
  end;
  for i := 0 to insn.childNodes.Count-1 do begin
    if nextState = '' then
      EvalInsn(outputPattern, nextState, inputPattern, state, insn.childNodes[i]);
  end;
end;

procedure TFSM.GenerateTransitions;
  procedure Worker(fromState, inputPattern : String);
    var outputPattern, nextState : String;
    var i : Integer;
  begin
    if length(inputPattern) < inputSignals.Count then begin
      Worker(fromState, inputPattern + '0');
      Worker(fromState, inputPattern + '1');
      exit;
    end;

    outputPattern := '';
    for i := 1 to outputSignals.Count do
      outputPattern := outputPattern + 'x';

    EvalInsn(outputPattern, nextState, inputPattern, fromState, defaultActions);
    if nextState = '' then
      EvalInsn(outputPattern, nextState, inputPattern, fromState, stateList[fromState]);
    if nextState = '' then
      nextState := fromState;

    transitions[fromState + ':' + inputPattern] := nextState + ':' + outputPattern;
  end;
  var stateIdx : Integer;
begin
  for stateIdx := 0 to stateList.Count-1 do
    Worker(stateList.Keys[stateIdx], '');
end;

procedure TFSM.CompressTransitions;
  var keepRunning : Boolean;
  var afterColon : Boolean;
  var i, j : Integer;
  var t1, t2 : String;
  var newTransitions : TTransitionList;
  label nextTransition;
begin
  keepRunning := true;
  while keepRunning do begin
    keepRunning := false;
    newTransitions := TTransitionList.Create;
    for i := 0 to transitions.Count-1 do begin
      t1 := transitions.Keys[i];
      if transitions[t1] = '' then
        goto nextTransition;
      afterColon := false;
      for j := 1 to length(t1) do begin
        if afterColon and ((t1[j] = '1') or (t1[j] = '0')) then begin
          t2 := t1;
          if t1[j] = '1' then
            t2[j] := '0';
          if t1[j] = '0' then
            t2[j] := '1';
          if (transitions.indexOf(t2) >= 0) and (transitions[t1] = transitions[t2]) then begin
            transitions[t2] := '';
            t2[j] := 'x';
            newTransitions[t2] := transitions[t1];
            keepRunning := true;
            goto nextTransition;
          end;
        end;
        if t1[j] = ':' then
          afterColon := true;
      end;
      newTransitions[t1] := transitions[t1];
      nextTransition:
    end;
    transitions.Free;
    transitions := newTransitions;
  end;
end;

procedure TFSM.CollectTransitions;
  var i, j : Integer;
  var t1, t2, t3, t4 : String;
  var reverseMap : TTransitionList;
  var copyState : Boolean;
begin
  reverseMap := TTransitionList.Create;
  for i := 0 to transitions.Count-1 do begin
    t1 := transitions.Keys[i];
    t2 := '';
    for j := 1 to length(t1) do begin
      t2 := t2 + t1[j];
      if t1[j] = ':' then Break;
    end;
    t2 := t2 + transitions[t1];
    if reverseMap.IndexOf(t2) >= 0 then
      reverseMap[t2] := reverseMap[t2] + '|' + t1
    else
      reverseMap[t2] := t1;
  end;
  transitions.Free;
  transitions := TTransitionList.Create;
  for i := 0 to reverseMap.Count-1 do begin
    t1 := reverseMap.Data[i];
    t2 := reverseMap.Keys[i];
    t3 := '';
    copyState := true;
    for j := 1 to length(t1) do begin
      if copyState then
        t3 := t3 + t1[j];
      if t1[j] = ':' then
        copyState := true;
      if t1[j] = '|' then
        copyState := false;
    end;
    t4 := '';
    copyState := false;
    for j := 1 to length(t2) do begin
      if copyState then
        t4 := t4 + t2[j];
      if t1[j] = ':' then
        copyState := true;
    end;
    transitions[t3] := t4;
  end;
  reverseMap.Free;
end;

constructor TSynth.Create(filename : String);
  var codefile : text;
  var i : Integer;
begin
  FSignalList := TSignalList.Create;
  FWordList := TSignalList.Create;
  FUnitInstances := TUnitInstanceList.Create;
  FFSMList := TFSMList.Create;

  assign(codefile, filename);
  reset(codefile);
  yyinput := codefile;
  yysynth := self;
  yyinsstack_idx := 0;
  yyparse;

  Worker_SubstNext;
  for i := 0 to FFSMList.Count-1 do begin
    FFSMList.Data[i].GenerateTransitions;
    FFSMList.Data[i].CompressTransitions;
    FFSMList.Data[i].CollectTransitions;
  end;
end;

function TSynth.CheckId(id : String) : Boolean;
begin
  CheckId := false;
  if FSignalList.IndexOf(id) >= 0 then
    CheckId := true;
  if FWordList.IndexOf(id) >= 0 then
    CheckId := true;
  if FUnitInstances.IndexOf(id) >= 0 then
    CheckId := true;
  if FFSMList.IndexOf(id) >= 0 then
    CheckId := true;
end;

procedure TSynth.Worker_SubstNext;
  procedure FindAndReplaceNext(insn : TFSMStateInsn; nextState : String);
    var k : Integer;
  begin
    if insn.insnType = 'NEXT' then begin
      insn.insnType := 'GOTO';
      insn.arg1 := nextState;
    end;
    for k := 0 to insn.childNodes.Count-1 do
      FindAndReplaceNext(insn.childNodes[k], nextState);
  end;
  var i, j : Integer;
  var id1, id2, id3 : String;
begin
  for i := 0 to FFSMList.Count-1 do begin
    id1 := FFSMList.Keys[i];
    for j := 0 to FFSMList[id1].stateList.Count-1 do begin
      id2 := FFSMList[id1].stateList.Keys[j];
      if FFSMList[id1].state2next.IndexOf(id2) >= 0 then begin
        id3 := FFSMList[id1].state2next[id2];
        FindAndReplaceNext(FFSMList[id1].stateList[id2], id3);
      end;
    end;
  end;
end;

procedure TSynth.Dump;
  var i, j : Integer;
  var id, id2 : String;
begin
  for i := 0 to FSignalList.Count-1 do begin
    id := FSignalList.Keys[i];
    Write('signal');
    if FSignalList[id].inputName > '' then
      Write(Format(' input(%s)', [FSignalList[id].inputName]));
    if FSignalList[id].outputName > '' then
      Write(Format(' output(%s)', [FSignalList[id].outputName]));
    WriteLn(' ', id, ';');
  end;

  for i := 0 to FWordList.Count-1 do begin
    id := FWordList.Keys[i];
    Write('word');
    if FWordList[id].inputName > '' then
      Write(Format(' input(%s)', [FWordList[id].inputName]));
    if FWordList[id].outputName  > '' then
      Write(Format(' output(%s)', [FWordList[id].outputName]));
    WriteLn(' ', id, ';');
  end;

  for i := 0 to FUnitInstances.Count-1 do begin
    id := FUnitInstances.Keys[i];
    WriteLn('unit ' + FUnitInstances[id].unitTypeName + ' ' + id + ' begin');
    for j := 0 to FUnitInstances[id].signalMap.Count-1 do
      WriteLn('  ' + FUnitInstances[id].signalMap.Keys[j] + ': '
                   + FUnitInstances[id].signalMap.Data[j] + ';');
    WriteLn('end');
  end;

  for i := 0 to FFSMList.Count-1 do begin
    id := FFSMList.Keys[i];
    WriteLn('fsm ' + id + ' begin');
    Write('  // inputs:');
    for j := 0 to FFSMList[id].inputSignals.Count-1 do
      Write(' ', FFSMList[id].inputSignals[j]);
    WriteLn;
    Write('  // outputs:');
    for j := 0 to FFSMList[id].outputSignals.Count-1 do
      Write(' ', FFSMList[id].outputSignals[j]);
    WriteLn;
    for j := 0 to FFSMList[id].transitions.Count-1 do
      WriteLn('  // TR: ', FFSMList[id].transitions.Keys[j], ' -> ', FFSMList[id].transitions.Data[j]);
    FFSMList[id].defaultActions.Dump('');
    for j := 0 to FFSMList[id].stateList.Count-1 do begin
      id2 := FFSMList[id].stateList.Keys[j];
      WriteLn(id2 + ':');
      FFSMList[id].stateList[id2].Dump('');
    end;
    WriteLn('end');
  end;
end;

var sobj : TSynth;
begin
  sobj := TSynth.Create('example1.txt');
  sobj.Dump;
  sobj.Free;
end.
