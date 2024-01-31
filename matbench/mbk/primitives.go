
package mbk

// ------------------------------------------------------------

type Atom interface {
	String() string;
	Eq(other Atom) bool;
	IsNil() bool;
	IsBool() *ABool;
	IsCons() *ACons;
	IsId() *AId;
	IsDict() *ADict;
	IsFunc() *AFunc;
	IsBuiltin() *ABuiltin;
}

type ABase struct {
}

func (this *ABase) String() string {
	return "*AtomWithoutStringRepresentation*";
}

func (this *ABase) Eq(other Atom) bool {
	return false;
}

func (this *ABase) IsNil() bool {
	return false;
}

func (this *ABase) IsBool() *ABool {
	return nil;
}

func (this *ABase) IsCons() *ACons {
	return nil;
}

func (this *ABase) IsId() *AId {
	return nil;
}

func (this *ABase) IsDict() *ADict {
	return nil;
}

func (this *ABase) IsFunc() *AFunc {
	return nil;
}

func (this *ABase) IsBuiltin() *ABuiltin {
	return nil;
}

// ------------------------------------------------------------

type ANil struct {
	ABase;
}

func NewANil() (this *ANil) {
	this = new(ANil);
	return;
}

func (this *ANil) String() string {
	return "NIL";
}

func (this *ANil) Eq(other Atom) bool {
	return other.IsNil();
}

func (this *ANil) IsNil() bool {
	return true;
}

// ------------------------------------------------------------

type ABool struct {
	ABase;
	v bool;
}

func NewABool(v bool) (this *ABool) {
	this = new(ABool);
	this.v = v;
	return;
}

func (this *ABool) True() bool {
	return this.v;
}

func (this *ABool) False() bool {
	return !this.v;
}

func (this *ABool) String() string {
	if this.v { return "TRUE"; }
	return "FALSE";
}

func (this *ABool) Eq(other Atom) bool {
	if o := other.IsBool(); o != nil && o.v == this.v {
		return true;
	}
	return false;
}

func (this *ABool) IsBool() *ABool {
	return this;
}


// ------------------------------------------------------------

type ACons struct {
	ABase;
	car, cdr Atom;
}

func NewACons(car, cdr Atom) (this *ACons) {
	this = new(ACons);
	this.car = car;
	this.cdr = cdr;
	return;
}

func (this *ACons) Car() Atom {
	return this.car;
}

func (this *ACons) Cdr() Atom {
	return this.cdr;
}

func (this *ACons) String() string {
	str := "(";
	delim := "";
	p := this;
	for {
		if _, cdr_nil_ok := p.cdr.(*ANil); cdr_nil_ok {
			str += delim + p.car.String();
			break;
		}
		cdr_cons, cdr_cons_ok := p.cdr.(*ACons);
		if !cdr_cons_ok {
			return "(" + this.car.String() + " . " + this.cdr.String() + ")";
		}
		str += delim + p.car.String();
		delim = " ";
		p = cdr_cons;
	}
	return str + ")";
}

func (this *ACons) Eq(other Atom) bool {
	if o := other.IsCons(); o == this {
		return true;
	}
	return false;
}

func (this *ACons) IsCons() *ACons {
	return this;
}

// ------------------------------------------------------------

type AId struct {
	ABase;
	id string;
}

func NewAId(id string) (this *AId) {
	this = new(AId);
	this.id = id;
	return;
}

func (this *AId) String() string {
	return this.id;
}

func (this *AId) Eq(other Atom) bool {
	if o := other.IsId(); o != nil && o.id == this.id {
		return true;
	}
	return false;
}

func (this *AId) IsId() *AId {
	return this;
}

// ------------------------------------------------------------

type ADict struct {
	ABase;
	elements map[string] Atom;
	parent *ADict;
}

func NewADict(parent *ADict) (this *ADict) {
	this = new(ADict);
	this.elements = make(map[string] Atom);
	this.parent = parent;
	return;
}

func (this *ADict) String() string {
	str := "{";
	delim := "";
	for key, value := range this.elements {
		str += delim + key + " " + value.String();
		delim = " ";
	}
	return str + "}";
}

func (this *ADict) Def(key Atom, value Atom) {
	this.elements[key.String()] = value;
}

func (this *ADict) Undef(key Atom) {
	this.elements[key.String()] = nil, false;
}

func (this *ADict) set_worker(key_s string, value Atom) bool {
	if _, ok := this.elements[key_s]; ok {
		this.elements[key_s] = value;
		return true;
	}
	if this.parent != nil {
		return this.parent.set_worker(key_s, value);
	}
	return false;
}

func (this *ADict) Set(key Atom, value Atom) {
	key_s := key.String();
	if this.set_worker(key_s, value) == false {
		this.elements[key_s] = value;
	}
}

func (this *ADict) get_worker(key_s string) Atom {
	if v, ok := this.elements[key_s]; ok {
		return v;
	}
	if this.parent != nil {
		return this.parent.get_worker(key_s);
	}
	return nil;
}

func (this *ADict) Get(key Atom) Atom {
	key_s := key.String();
	return this.get_worker(key_s);
}

func (this *ADict) Eq(other Atom) bool {
	if o := other.IsDict(); o == this {
		return true;
	}
	return false;
}

func (this *ADict) IsDict() *ADict {
	return this;
}

// ------------------------------------------------------------

type AFunc struct {
	ABase;
	code Atom;
	context *ADict;
}

func NewAFunc(code Atom, context *ADict) (this *AFunc) {
	this = new(AFunc);
	this.code = code;
	this.context = context;
	return;
}

func (this *AFunc) String() string {
	return "#" + this.code.String();
}

func (this *AFunc) Eq(other Atom) bool {
	if o := other.IsFunc(); o == this {
		return true;
	}
	return false;
}

func (this *AFunc) IsFunc() *AFunc {
	return this;
}

// ------------------------------------------------------------

type ArgsCtrl struct {
	c *ACons;
	context *ADict;
	special *ADict;
}

func (this *ArgsCtrl) Context() (*ADict) {
	return this.context;
}

func (this *ArgsCtrl) Special() (*ADict) {
	return this.special;
}

func (this *ArgsCtrl) QuoteNext() (a, j Atom) {
	if this.c == nil {
		return NewANil(), NewAId("unexpected-end-of-arguments-list-exception");
	}
	a = this.c.Car();
	this.c = this.c.Cdr().IsCons();
	return a, j;
}

func (this *ArgsCtrl) EvalNext() (a, j Atom) {
	if this.c == nil {
		return NewANil(), NewAId("unexpected-end-of-arguments-list-exception");
	}
	a, j = Execute(this.c.Car(), this.context, this.special);
	this.c = this.c.Cdr().IsCons();
	return a, j;
}

func (this *ArgsCtrl) SkipNext() {
	if this.c != nil {
		this.c = this.c.Cdr().IsCons();
	}
}

func (this *ArgsCtrl) Last() bool {
	return this.c == nil;
}

type ABuiltinHandler func (args *ArgsCtrl) (r, jmp Atom);

type ABuiltin struct {
	ABase;
	handler ABuiltinHandler;
}

func NewABuiltin(handler ABuiltinHandler) (this *ABuiltin) {
	this = new(ABuiltin);
	this.handler = handler;
	return;
}

func (this *ABuiltin) String() string {
	return "#(builtin)";
}

func (this *ABuiltin) Eq(other Atom) bool {
	if o := other.IsBuiltin(); o == this {
		return true;
	}
	return false;
}

func (this *ABuiltin) IsBuiltin() *ABuiltin {
	return this;
}

