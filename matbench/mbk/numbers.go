
package mbk
import "fmt"
import "big"
import "math"

const (
	NUMTYPE_Q = iota;
	NUMTYPE_R = iota;
	NUMTYPE_CQ = iota;
	NUMTYPE_CR = iota;
	NUMTYPE_NONE = iota;
)

type ANumQ struct {
	ABase;
	n, m *big.Int;
}

type ANumR struct {
	ABase;
	n float64;
}

func NewANumber(t string) (a Atom) {
	neg := false;
	var decimal float64 = 0.0;
	var f float64 = 0.0;
	i := 0;
	n := big.NewInt(0);
	ten := big.NewInt(10);
	for; i < len(t); i++ {
		ch := t[i];
		if ch == '/' {
			break;
		}
		if ch == '-' {
			neg = !neg;
		}
		if ch == '.' {
			decimal = 1;
		}
		if ch >= '0' && ch <= '9' {
			n.Mul(n, ten);
			n.Add(n, big.NewInt(int64(ch - '0')));
			if decimal > 0 {
				decimal *= 10;
				f += float64(ch - '0') / decimal;
			} else {
				f = f*10 + float64(ch - '0');
			}
		}
	}
	if neg {
		f *= -1;
		n.Mul(n, big.NewInt(-1));
	}
	if decimal > 0 {
		a = &ANumR{n: f};
		return a;
	}
	if i < len(t) && t[i] == '/' {
		m := big.NewInt(0);
		for; i < len(t); i++ {
			ch := t[i];
			if ch >= '0' && ch <= '9' {
				m.Mul(m, ten);
				m.Add(m, big.NewInt(int64(ch - '0')));
			}
		}
		a = &ANumQ{n: n, m: m};
		return a;
	}
	a = &ANumQ{n: n, m: big.NewInt(1)};
	return a;
}

func bigIntToFloat(n *big.Int) (f float64) {
	b := n.Bytes();
	var fact float64 = 1.0;
	for i := len(b)-1; i >= 0; i-- {
		f += float64(b[i]) * fact;
		fact *= 256;
	}
	return;
}

func (this *ANumQ) String() string {
	n, m := this.n.String(), this.m.String();
	if m == "1" {
		return n;
	}
	return n + "/" + m;
}

func (this *ANumR) String() string {
	return fmt.Sprintf("%f", this.n);
}

// ------------------------------------------------------------

type NumFuncQ func(an, am []*big.Int) (r, jmp Atom);
type NumFuncR func(af []float64) (r, jmp Atom);

func EvalNumberFunc(args *ArgsCtrl, op string, num int, fq NumFuncQ, fr NumFuncR) (r, jmp Atom) {
	tp := NUMTYPE_Q;

	var el = make([]Atom, 0, 100);
	for !args.Last() {
		n := len(el);
		el = el[0:n+1];
		el[n], jmp = args.EvalNext();
		if jmp != nil {
			r = el[n];
			return;
		}
		switch tp {
		case NUMTYPE_Q:
			if _, ok := el[n].(*ANumQ); ok {
				/* nothing */
			} else if _, ok := el[n].(*ANumR); ok {
				tp = NUMTYPE_R;
			} else {
				tp = NUMTYPE_NONE;
			}
		case NUMTYPE_R:
			if _, ok := el[n].(*ANumQ); ok {
				/* nothing */
			} else if _, ok := el[n].(*ANumR); ok {
				/* nothing */
			} else {
				tp = NUMTYPE_NONE;
			}
		}
	}

	if len(el) < num {
		r = NewANil();
		jmp = NewAId("called-math-function-with-invalid-number-of-arguments-exception");
		return;
	}

	if tp == NUMTYPE_NONE {
		r = NewANil();
		for i := len(el)-1; i >= 0; i-- {
			r = NewACons(el[i], r);
		}
		r = NewACons(NewAId(op), r);
		return;
	}

	switch tp {
	case NUMTYPE_Q:
		an := make([]*big.Int, len(el));
		am := make([]*big.Int, len(el));
		for i := 0; i < len(el); i++ {
			if a, ok := el[i].(*ANumQ); ok {
				an[i] = a.n;
				am[i] = a.m;
			}
		}
		r, jmp = fq(an, am);
		if rq, ok := r.(*ANumQ); ok {
			n := big.NewInt(0);
			big.GcdInt(n, nil, nil, rq.n, rq.m);
			rq.n.Div(rq.n, n);
			rq.m.Div(rq.m, n);
		}
	case NUMTYPE_R:
		af := make([]float64, len(el));
		for i := 0; i < len(el); i++ {
			if a, ok := el[i].(*ANumQ); ok {
				af[i] = bigIntToFloat(a.n) / bigIntToFloat(a.m);
			} else if a, ok := el[i].(*ANumR); ok {
				af[i] = a.n;
			}
		}
		r, jmp = fr(af);
	}

	return;
}

// ------------------------------------------------------------

func builtin_num_addQ(an, am []*big.Int) (a, jmp Atom) {
	r := &ANumQ{n: big.NewInt(0), m: big.NewInt(1)};
	for i := 0; i < len(an); i++ {
		n := big.NewInt(0);
		n.Mul(r.m, an[i]);
		r.n.Mul(r.n, am[i]);
		r.n.Add(r.n, n);
		r.m.Mul(am[i], r.m);
	}
	return r, jmp;
}

func builtin_num_addR(af []float64) (a, jmp Atom) {
	r := &ANumR{n: 0.0};
	for i := 0; i < len(af); i++ {
		r.n += af[i];
	}
	return r, jmp;
}

func builtin_num_add(args *ArgsCtrl) (r, jmp Atom) {
	r, jmp = EvalNumberFunc(args, "+", 0, builtin_num_addQ, builtin_num_addR);
	return;
}

// ------------------------------------------------------------

func builtin_num_subQ(an, am []*big.Int) (a, jmp Atom) {
	r := &ANumQ{n: big.NewInt(0), m: big.NewInt(1)};
	for i := 0; i < len(an); i++ {
		n := big.NewInt(0);
		n.Mul(r.m, an[i]);
		r.n.Mul(r.n, am[i]);
		if i == 0 && len(an) > 1 {
			r.n.Add(r.n, n);
		} else {
			r.n.Sub(r.n, n);
		}
		r.m.Mul(am[i], r.m);
	}
	return r, jmp;
}

func builtin_num_subR(af []float64) (a, jmp Atom) {
	r := &ANumR{n: 0.0};
	for i := 0; i < len(af); i++ {
		if i == 0 && len(af) > 1 {
			r.n += af[i];
		} else {
			r.n -= af[i];
		}
	}
	return r, jmp;
}

func builtin_num_sub(args *ArgsCtrl) (r, jmp Atom) {
	r, jmp = EvalNumberFunc(args, "-", 0, builtin_num_subQ, builtin_num_subR);
	return;
}

// ------------------------------------------------------------

func builtin_num_mulQ(an, am []*big.Int) (a, jmp Atom) {
	r := &ANumQ{n: big.NewInt(1), m: big.NewInt(1)};
	for i := 0; i < len(an); i++ {
		r.n.Mul(r.n, an[i]);
		r.m.Mul(r.m, am[i]);
	}
	return r, jmp;
}

func builtin_num_mulR(af []float64) (a, jmp Atom) {
	r := &ANumR{n: 1.0};
	for i := 0; i < len(af); i++ {
		r.n *= af[i];
	}
	return r, jmp;
}

func builtin_num_mul(args *ArgsCtrl) (r, jmp Atom) {
	r, jmp = EvalNumberFunc(args, "*", 0, builtin_num_mulQ, builtin_num_mulR);
	return;
}

// ------------------------------------------------------------

func builtin_num_divQ(an, am []*big.Int) (a, jmp Atom) {
	r := &ANumQ{n: big.NewInt(1), m: big.NewInt(1)};
	for i := 0; i < len(an); i++ {
		if i == 0 && len(an) > 1 {
			r.n.Mul(r.n, an[i]);
			r.m.Mul(r.m, am[i]);
		} else {
			r.n.Mul(r.n, am[i]);
			r.m.Mul(r.m, an[i]);
		}
	}
	return r, jmp;
}

func builtin_num_divR(af []float64) (a, jmp Atom) {
	r := &ANumR{n: 1.0};
	for i := 0; i < len(af); i++ {
		if i == 0 && len(af) > 1 {
			r.n *= af[i];
		} else {
			r.n /= af[i];
		}
	}
	return r, jmp;
}

func builtin_num_div(args *ArgsCtrl) (r, jmp Atom) {
	r, jmp = EvalNumberFunc(args, "/", 0, builtin_num_divQ, builtin_num_divR);
	return;
}

// ------------------------------------------------------------

func builtin_num_modQ(an, am []*big.Int) (a, jmp Atom) {
	if am[0].String() == "1" && am[1].String() == "1" {
		r := &ANumQ{n: big.NewInt(1), m: big.NewInt(1)};
		r.n.Mod(an[0], an[1]);
		return r, jmp;
	}
	// FIXME
	x := bigIntToFloat(an[0]) / bigIntToFloat(am[0]);
	y := bigIntToFloat(an[1]) / bigIntToFloat(am[1]);
	return &ANumR{n: math.Fmod(x, y)}, jmp;
}

func builtin_num_modR(af []float64) (a, jmp Atom) {
	return &ANumR{n: math.Fmod(af[0], af[1])}, jmp;
}

func builtin_num_mod(args *ArgsCtrl) (r, jmp Atom) {
	r, jmp = EvalNumberFunc(args, "%", 2, builtin_num_modQ, builtin_num_modR);
	return;
}


// ------------------------------------------------------------

func builtin_num_powQ(an, am []*big.Int) (a, jmp Atom) {
	if am[1].String() == "1" {
		r := &ANumQ{n: big.NewInt(1), m: big.NewInt(1)};
		r.n.Exp(an[0], an[1], nil);
		r.m.Exp(am[0], an[1], nil);
		return r, jmp;
	}
	x := bigIntToFloat(an[0]) / bigIntToFloat(am[0]);
	y := bigIntToFloat(an[1]) / bigIntToFloat(am[1]);
	return &ANumR{n: math.Pow(x, y)}, jmp;
}

func builtin_num_powR(af []float64) (a, jmp Atom) {
	return &ANumR{n: math.Pow(af[0], af[1])}, jmp;
}

func builtin_num_pow(args *ArgsCtrl) (r, jmp Atom) {
	r, jmp = EvalNumberFunc(args, "**", 2, builtin_num_powQ, builtin_num_powR);
	return;
}

// ------------------------------------------------------------

func builtin_num_floorQ(an, am []*big.Int) (a, jmp Atom) {
	r := &ANumQ{n: big.NewInt(1), m: big.NewInt(1)};
	r.n.Div(an[0], am[0]);
	return r, jmp;
}

func builtin_num_floorR(af []float64) (a, jmp Atom) {
	return &ANumQ{n: big.NewInt(int64(math.Floor(af[0]))), m: big.NewInt(1)}, jmp;
}

func builtin_num_floor(args *ArgsCtrl) (r, jmp Atom) {
	r, jmp = EvalNumberFunc(args, "floor", 1, builtin_num_floorQ, builtin_num_floorR);
	return;
}

// ------------------------------------------------------------

func builtin_num_maxQ(an, am []*big.Int) (a, jmp Atom) {
	r := &ANumQ{n: big.NewInt(1), m: big.NewInt(1)};
	r.n.Mul(an[0], big.NewInt(+1));
	r.m.Mul(am[0], big.NewInt(+1));

	for i := 1; i < len(an); i++ {
		n := big.NewInt(1);
		r.n.Mul(r.n, am[i]);
		r.m.Mul(r.m, am[i]);
		n.Mul(r.m, an[i]);
		if n.Cmp(r.n) > 0 {
			r.n = n;
		}
	}

	return r, jmp;
}

func builtin_num_maxR(af []float64) (a, jmp Atom) {
	f := af[0];
	for i := 1; i < len(af); i++ {
		if af[i] > f {
			f = af[i];
		}
	}
	return &ANumR{n: f}, jmp;
}

func builtin_num_max(args *ArgsCtrl) (r, jmp Atom) {
	r, jmp = EvalNumberFunc(args, "max", 1, builtin_num_maxQ, builtin_num_maxR);
	return;
}

// ------------------------------------------------------------

func builtin_num_eqQ(an, am []*big.Int) (a, jmp Atom) {
	return NewABool(an[0].Cmp(an[1]) == 0 && am[0].Cmp(am[1]) == 0), jmp;
}

func builtin_num_eqR(af []float64) (a, jmp Atom) {
	return NewABool(af[0] == af[1]), jmp;
}

func builtin_num_eq(args *ArgsCtrl) (r, jmp Atom) {
	r, jmp = EvalNumberFunc(args, "=", 2, builtin_num_eqQ, builtin_num_eqR);
	return;
}

// ------------------------------------------------------------

func RegisterNumberBuiltins(context *ADict) {
	context.Def(NewAId("+"), NewABuiltin(builtin_num_add));
	context.Def(NewAId("-"), NewABuiltin(builtin_num_sub));
	context.Def(NewAId("*"), NewABuiltin(builtin_num_mul));
	context.Def(NewAId("/"), NewABuiltin(builtin_num_div));
	context.Def(NewAId("%"), NewABuiltin(builtin_num_mod));
	context.Def(NewAId("**"), NewABuiltin(builtin_num_pow));

	context.Def(NewAId("floor"), NewABuiltin(builtin_num_floor));
	context.Def(NewAId("max"), NewABuiltin(builtin_num_max));

	context.Def(NewAId("="), NewABuiltin(builtin_num_eq));
}

