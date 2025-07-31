from dataclasses import dataclass, field
from z3 import Solver, sat, unsat, simplify, substitute
from z3 import Bool, Not, And, Or, Xor, Implies
import parser

def collapse_args(tokens):
    """
    Collapse each top-level (...) span into a sublist that includes the outer '(' and ')'.
    Assumes balanced parentheses.
    """
    result = []
    depth = 0
    current = None

    for tok in tokens:
        if tok == '(':
            depth += 1
            if depth == 1:
                current = ['(']
            else:
                current.append('(')
        elif tok == ')':
            if depth == 0:
                raise ValueError("Unbalanced ')'")
            if depth == 1:
                current.append(')')
                result.append(current)
                current = None
            else:
                current.append(')')
            depth -= 1
        else:
            if depth == 0:
                result.append(tok)
            else:
                current.append(tok)

    if depth != 0:
        raise ValueError("Unbalanced '('")
    return result

class FormalModel:
    def __init__(self, code, name=None, blocks=None):
        if isinstance(code, str):
            code = parser.Parser(code).code
        if name is None:
            name = (*code.names,)[-1]
        blocks = set(blocks) if blocks else set()
        self.syms = dict()
        self.code = code
        self.name = name
        first = code.names[name][""][0]
        stmts = code.stmts[first:]

        for i, s in enumerate(stmts):
            if i and s.tokens[0] in ('ENDMODULE', 'MODULE'):
                stmts = stmts[:i]
                break

        en = True
        self.stmts = []
        self.blocks = {}
        for s in stmts:
            if s.tokens[0] in ('TABLE', 'PTABLE', 'CIRCUIT'):
                key = s.tokens[1].removeprefix('"').removesuffix('"')
                if blocks:
                    if en := (key in blocks):
                        self.blocks[key] = len(self.stmts)
                else:
                    self.blocks[key] = len(self.stmts)
            if en:
                self.stmts.append(s)

    def sym(self, name):
        try:
            return self.syms[name]
        except KeyError:
            val = Bool(name)
            self.syms[name] = val
            return val

    def expr(self, toks, spec):
        if isinstance(toks, str):
            s = self.sym(toks)
            if s not in spec.outputs:
                spec.inputs.add(s)
            return s
        if len(toks) == 1 and toks[0][0] in "iqnod":
            return self.sym(toks[0])

        if toks[0] == "(" and toks[-1] == ")":
            op = toks[1]
            args = [self.expr(t, spec) for t in collapse_args(toks[2:-1])]

            if op == "NOT":
                assert len(args) == 1
                return Not(args[0])

            if op == "AND":
                assert len(args) > 0
                return And(*args)

            if op == "NAND":
                assert len(args) > 0
                return Not(And(*args))

            if op == "OR":
                assert len(args) > 0
                return Or(*args)

            if op == "NOR":
                assert len(args) > 0
                return Not(Or(*args))

            assert False, f"Unsupported OP: {op} (with {len(args)} args)"

        assert False, f"Invalid Expr: {toks}"

    def block(self, name):
        spec = FormalSpec(self)
        idx = self.blocks[name]
        hdr = self.stmts[idx]
        idx += 1

        assert hdr.tokens[0] not in ('TABLE', 'PTABLE')

        while idx < len(self.stmts):
            s = self.stmts[idx]; idx += 1
            if s.tokens[0] in ('BREAK',):
                continue
            if s.tokens[0] in ('TABLE', 'PTABLE', 'CIRCUIT'):
                break

            if s.tokens[0] == "DEF" and s.tokens[1][0] in "nod":
                lhs = self.sym(s.tokens[1])
                rhs = self.expr(s.tokens[2:], spec)
                spec.inputs.discard(lhs)
                spec.outputs[lhs] = simplify(rhs)
                continue

            assert False, f"Block Parser Error at {s.tokens[0]}"

        keep_running = True
        while keep_running:
            keep_running = False
            for key in spec.outputs.keys():
                old_rhs = spec.outputs[key]
                new_rhs = simplify(substitute(old_rhs, *spec.outputs.items()))
                if str(old_rhs) != str(new_rhs):
                    spec.outputs[key] = new_rhs
                    keep_running = True

        return spec

@dataclass
class FormalSpec:
    model: FormalModel
    inputs: set = field(default_factory=set)
    outputs: dict = field(default_factory=dict)

    def check(self, other):
        keys = set(self.output.keys()).intersection(set(other.output.keys()))
        terms = [self.output[key] == other.output[key] for key in keys]
        return And(*terms)

if __name__ == "__main__":
    fm = FormalModel("""
MODULE "demo"
  DIMS i1 q0 n0 o0 d0 f0 a0
  PI i0 i1
  PO o0

  CIRCUIT "gold"
    DEF o0 (AND i1 i2)

  TABLE "tab"
    GET i0 i1
    SET o0
    DEF '11 ==> 1'
    DEF '0- ==> 0'
    DEF '-0 ==> 0'

  PTABLE "ptab"
    GET i0 i1
    SET o0
    DEF '11 ==> 1'
    DEF '-- ==> 0'

  CIRCUIT "gate1"
    DEF o0 (NOR (NOT i1) (NOT i2))

  CIRCUIT "gate2"
    DEF o0 (NOR (NOT i1) i2)

  CIRCUIT "gate3"
    DEF n0 (OR (NOT i1) (NOT i2))
    DEF o0 (NOT n0)
ENDMODULE
""")
    gold = fm.block("gold")
    print(gold)
    for g in ("gate1", "gate2", "gate3"):
        gate = fm.block(g)
        print(gate)
        #s = Solver()
        #s.add(Not(Implies(gold, gate)))
        #print(f"\ngold vs {g}:")
        #if s.check() == sat:
        #    print(s.model())
        #else:
        #    print("unsat")
    print()
