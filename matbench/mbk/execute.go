
package mbk

func Execute(code Atom, context *ADict, special *ADict) (r, jmp Atom) {
	c := code.IsCons();
	if c == nil {
		if id := code.IsId(); id != nil {
			if id.String()[0] == '$' {
				if v := special.Get(id); v != nil {
					return v, jmp;
				}
			} else
			if v := context.Get(id); v != nil {
				return v, jmp;
			}
		}
		return code, jmp;
	}

	eval_next := func() (a, j Atom) {
		if c == nil {
			return NewANil(), NewAId("unexpected-end-of-arguments-list-exception");
		}
		a, j = Execute(c.Car(), context, special);
		c = c.Cdr().IsCons();
		return a, j;
	};

	ins, jmp := eval_next();
	if jmp != nil {
		return ins, jmp;
	}

	if ins_id := ins.IsId(); ins_id != nil {
		builtin := ins_id.String();
		if builtin[0] == 'c' {
			steps := 0;
			for steps+2 < len(builtin) && (builtin[1+steps] == 'a' || builtin[1+steps] == 'd') { steps++; }
			if steps+2 == len(builtin) && builtin[1+steps] == 'r' {
				v, jmp := eval_next();
				if jmp != nil {
					return v, jmp;
				}
				for i := 0; i < steps; i++ {
					if vc := v.IsCons(); vc != nil {
						if builtin[1+steps] == 'a' {
							v = vc.Car();
						} else {
							v = vc.Cdr();
						}
					} else if !v.IsNil() {
						return NewANil(), NewAId("car-cdr-on-non-cons-or-nil-exception");
					}
				}
				return v, jmp;
			}
		}
		return NewANil(), NewAId("not-a-function-exception");
	} else if f := ins.IsFunc(); f != nil {
		var el = make([]Atom, 0, 100);
		for c != nil {
			n := len(el);
			el = el[0:n+1];
			el[n], jmp = eval_next();
			if jmp != nil {
				return el[n], jmp;
			}
		}

		var new_dd Atom = NewANil();
		for i := len(el)-1; i >= 0; i-- {
			new_dd = NewACons(el[i], new_dd);
		}

		id_dd := NewAId("$$");
		backup_dd := special.Get(id_dd);
		special.Set(id_dd, new_dd);

		r, jmp = Execute(f.code, f.context, special);

		special.Set(id_dd, backup_dd);
	} else if f := ins.IsBuiltin(); f != nil {
		ac := &ArgsCtrl{ c, context, special };
		r, jmp = f.handler(ac);
	} else {
		r, jmp = NewANil(), NewAId("not-a-function-exception");
	}

	return r, jmp;
}

