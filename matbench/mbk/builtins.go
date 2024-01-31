
package mbk
import "fmt"

func builtin_quote(args *ArgsCtrl) (r, jmp Atom) {
	r, jmp = args.QuoteNext();
	return r, jmp;
}

func builtin_def(args *ArgsCtrl) (r, jmp Atom) {
	n, jmp := args.EvalNext();
	if jmp != nil {
		return n, jmp;
	}
	v, jmp := args.EvalNext();
	if jmp != nil {
		return v, jmp;
	}
	if n.String()[0] == '$' {
		// FIXME
		args.Special().Def(n, v);
	} else {
		args.Context().Def(n, v);
	}
	return v, jmp;
}

func builtin_set(args *ArgsCtrl) (r, jmp Atom) {
	n, jmp := args.EvalNext();
	if jmp != nil {
		return n, jmp;
	}
	v, jmp := args.EvalNext();
	if jmp != nil {
		return v, jmp;
	}
	if n.String()[0] == '$' {
		// FIXME
		args.Special().Set(n, v);
	} else {
		args.Context().Set(n, v);
	}
	return v, jmp;
}

func builtin_undef(args *ArgsCtrl) (r, jmp Atom) {
	n, jmp := args.EvalNext();
	if jmp != nil {
		return n, jmp;
	}
	if n.String()[0] == '$' {
		// FIXME
		args.Special().Undef(n);
	} else {
		args.Context().Undef(n);
	}
	return NewANil(), jmp;
}

func builtin_args(args *ArgsCtrl) (r, jmp Atom) {
	id_dd := NewAId("$$");
	dd := args.Special().Get(id_dd);
	for !args.Last() {
		n, jmp := args.EvalNext();
		if jmp != nil {
			return n, jmp;
		}
		var v Atom;
		if arg := dd.IsCons(); arg != nil {
			v = arg.Car();
			dd = arg.Cdr();
		} else {
			v = NewANil();
		}
		if n.String()[0] == '$' {
			// FIXME
			args.Special().Def(n, v);
		} else {
			args.Context().Def(n, v);
		}
	}
	args.Special().Set(id_dd, dd);
	return NewANil(), jmp;
}

func builtin_local(args *ArgsCtrl) (r, jmp Atom) {
	args.context = NewADict(args.Context());
	var e Atom = NewANil();
	for !args.Last() {
		e, jmp = args.EvalNext();
		if jmp != nil {
			return e, jmp;
		}
	}
	return e, jmp;
}

func builtin_namespace(args *ArgsCtrl) (r, jmp Atom) {
	args.context = NewADict(args.Context());
	var e Atom = NewANil();
	for !args.Last() {
		e, jmp = args.EvalNext();
		if jmp != nil {
			return e, jmp;
		}
	}
	return args.Context(), jmp;
}

func builtin_group(args *ArgsCtrl) (r, jmp Atom) {
	var e Atom = NewANil();
	for !args.Last() {
		e, jmp = args.EvalNext();
		if jmp != nil {
			return e, jmp;
		}
	}
	return e, jmp;
}

func builtin_loop(args *ArgsCtrl) (r, jmp Atom) {
	backup_c := args.c;
	for {
		for !args.Last() {
			e, jmp := args.EvalNext();
			if jmp != nil {
				return e, jmp;
			}
		}
		args.c = backup_c;
	}
	return;
}

func builtin_block(args *ArgsCtrl) (r, jmp Atom) {
	l, jmp := args.EvalNext();
	if jmp != nil {
		return l, jmp;
	}
	var e Atom = NewANil();
	for !args.Last() {
		e, jmp = args.EvalNext();
		if jmp != nil {
			if l.Eq(jmp) {
				return e, nil;
			}
			return e, jmp;
		}
	}
	return e, jmp;
}

func builtin_return(args *ArgsCtrl) (r, jmp Atom) {
	v, jmp := args.EvalNext();
	if jmp != nil {
		return v, jmp;
	}
	var l Atom = NewANil();
	if !args.Last() {
		l, jmp = args.EvalNext();
		if jmp != nil {
			return l, jmp;
		}
	}
	return v, l;
}

func builtin_first(args *ArgsCtrl) (r, jmp Atom) {
	v, jmp := args.EvalNext();
	if jmp != nil {
		return v, jmp;
	}
	if vc := v.IsCons(); vc != nil {
		return vc.Car(), jmp;
	} else if !v.IsNil() {
		return NewANil(), NewAId("argument-type-is-not-cons-or-nil-exception");
	}
	return NewANil(), jmp;
}

func builtin_rest(args *ArgsCtrl) (r, jmp Atom) {
	v, jmp := args.EvalNext();
	if jmp != nil {
		return v, jmp;
	}
	if vc := v.IsCons(); vc != nil {
		return vc.Cdr(), jmp;
	} else if !v.IsNil() {
		return NewANil(), NewAId("argument-type-is-not-cons-or-nil-exception");
	}
	return NewANil(), jmp;
}

func builtin_cons(args *ArgsCtrl) (r, jmp Atom) {
	car, jmp := args.EvalNext();
	if jmp != nil {
		return car, jmp;
	}
	cdr, jmp := args.EvalNext();
	if jmp != nil {
		return cdr, jmp;
	}
	return NewACons(car, cdr), jmp;
}

func builtin_if(args *ArgsCtrl) (r, jmp Atom) {
	cond, jmp := args.EvalNext();
	if jmp != nil {
		return cond, jmp;
	}
	if bc := cond.IsBool(); bc != nil {
		if bc.False() && !args.Last() {
			args.SkipNext();
		}
		branch, jmp := args.EvalNext();
		return branch, jmp;
	} else {
		return NewANil(), NewAId("argument-type-is-not-boolean-exception");
	}
	return NewANil(), jmp;
}

func builtin_when(args *ArgsCtrl) (r, jmp Atom) {
	cond, jmp := args.EvalNext();
	if jmp != nil {
		return cond, jmp;
	}
	if bc := cond.IsBool(); bc != nil {
		if bc.True() {
			var e Atom = NewANil();
			for !args.Last() {
				e, jmp = args.EvalNext();
				if jmp != nil {
					return e, jmp;
				}
			}
			return e, jmp;
		}
	} else {
		return NewANil(), NewAId("argument-type-is-not-boolean-exception");
	}
	return NewANil(), jmp;
}

func builtin_unless(args *ArgsCtrl) (r, jmp Atom) {
	cond, jmp := args.EvalNext();
	if jmp != nil {
		return cond, jmp;
	}
	if bc := cond.IsBool(); bc != nil {
		if bc.False() {
			var e Atom = NewANil();
			for !args.Last() {
				e, jmp = args.EvalNext();
				if jmp != nil {
					return e, jmp;
				}
			}
			return e, jmp;
		}
	} else {
		return NewANil(), NewAId("argument-type-is-not-boolean-exception");
	}
	return NewANil(), jmp;
}

func builtin_function(args *ArgsCtrl) (r, jmp Atom) {
	e, jmp := args.EvalNext();
	if jmp != nil {
		return e, jmp;
	}
	return NewAFunc(e, args.Context()), jmp;
}

func builtin_eval(args *ArgsCtrl) (r, jmp Atom) {
	// FIXME
	return NewANil(), jmp;
}

func builtin_parse(args *ArgsCtrl) (r, jmp Atom) {
	// FIXME
	return NewANil(), jmp;
}

func builtin_read(args *ArgsCtrl) (r, jmp Atom) {
	// FIXME
	return NewANil(), jmp;
}

func builtin_print(args *ArgsCtrl) (r, jmp Atom) {
	delim := "";
	output := "";
	for !args.Last() {
		e, jmp := args.EvalNext();
		if jmp != nil {
			return e, jmp;
		}
		output += fmt.Sprintf("%s%s", delim, e);
		delim = " ";
	}
	fmt.Printf("%s\n", output);
	return NewANil(), jmp;
}

func builtin_is_boolean(args *ArgsCtrl) (r, jmp Atom) {
	v, jmp := args.EvalNext();
	if jmp != nil {
		return v, jmp;
	}
	return NewABool(v.IsBool() != nil), jmp;
}

func builtin_is_id(args *ArgsCtrl) (r, jmp Atom) {
	v, jmp := args.EvalNext();
	if jmp != nil {
		return v, jmp;
	}
	return NewABool(v.IsBool() != nil), jmp;
}

func builtin_is_nil(args *ArgsCtrl) (r, jmp Atom) {
	v, jmp := args.EvalNext();
	if jmp != nil {
		return v, jmp;
	}
	return NewABool(v.IsNil()), jmp;
}

func builtin_is_list(args *ArgsCtrl) (r, jmp Atom) {
	v, jmp := args.EvalNext();
	if jmp != nil {
		return v, jmp;
	}
	return NewABool(v.IsCons() != nil), jmp;
}

func builtin_is_dict(args *ArgsCtrl) (r, jmp Atom) {
	v, jmp := args.EvalNext();
	if jmp != nil {
		return v, jmp;
	}
	return NewABool(v.IsDict() != nil), jmp;
}

func builtin_is_function(args *ArgsCtrl) (r, jmp Atom) {
	v, jmp := args.EvalNext();
	if jmp != nil {
		return v, jmp;
	}
	return NewABool(v.IsFunc() != nil || v.IsBuiltin() != nil), jmp;
}

func builtin_eq(args *ArgsCtrl) (r, jmp Atom) {
	v1, jmp := args.EvalNext();
	if jmp != nil {
		return v1, jmp;
	}
	v2, jmp := args.EvalNext();
	if jmp != nil {
		return v2, jmp;
	}
	return NewABool(v1.Eq(v2)), jmp;
}

func RegisterBuiltins(context *ADict) {
	context.Def(NewAId("quote"),       NewABuiltin(builtin_quote));
	context.Def(NewAId("def"),         NewABuiltin(builtin_def));
	context.Def(NewAId("set"),         NewABuiltin(builtin_set));
	context.Def(NewAId("undef"),       NewABuiltin(builtin_undef));
	context.Def(NewAId("args"),        NewABuiltin(builtin_args));
	context.Def(NewAId("local"),       NewABuiltin(builtin_local));
	context.Def(NewAId("namespace"),   NewABuiltin(builtin_namespace));
	context.Def(NewAId("group"),       NewABuiltin(builtin_group));
	context.Def(NewAId("loop"),        NewABuiltin(builtin_loop));
	context.Def(NewAId("block"),       NewABuiltin(builtin_block));
	context.Def(NewAId("return"),      NewABuiltin(builtin_return));
	context.Def(NewAId("first"),       NewABuiltin(builtin_first));
	context.Def(NewAId("rest"),        NewABuiltin(builtin_rest));
	context.Def(NewAId("cons"),        NewABuiltin(builtin_cons));
	context.Def(NewAId("if"),          NewABuiltin(builtin_if));
	context.Def(NewAId("when"),        NewABuiltin(builtin_when));
	context.Def(NewAId("unless"),      NewABuiltin(builtin_unless));
	context.Def(NewAId("function"),    NewABuiltin(builtin_function));
	context.Def(NewAId("eval"),        NewABuiltin(builtin_eval));
	context.Def(NewAId("read"),        NewABuiltin(builtin_read));
	context.Def(NewAId("print"),       NewABuiltin(builtin_print));
	context.Def(NewAId("is-boolean"),  NewABuiltin(builtin_is_boolean));
	context.Def(NewAId("is-id"),       NewABuiltin(builtin_is_id));
	context.Def(NewAId("is-nil"),      NewABuiltin(builtin_is_nil));
	context.Def(NewAId("is-list"),     NewABuiltin(builtin_is_list));
	context.Def(NewAId("is-dict"),     NewABuiltin(builtin_is_dict));
	context.Def(NewAId("is-function"), NewABuiltin(builtin_is_function));
	context.Def(NewAId("eq"),          NewABuiltin(builtin_eq));
}

func RegisterAllBuiltins(context *ADict) {
	RegisterBuiltins(context);
	RegisterNumberBuiltins(context);
}

