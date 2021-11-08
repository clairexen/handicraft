#!/usr/bin/env python3

class FNode:
    def __init__(self, fun, *args):
        self.fun = fun
        self.args = args

        if len(self.args) == 0:
            assert fun not in ("BUF", "NOT", "AND", "OR", "XOR", "MUX")

        if len(self.args) == 1:
            assert fun in ("BUF", "NOT")

        if len(self.args) == 2:
            assert fun in ("AND", "OR", "XOR")

        if len(self.args) == 3:
            assert fun in ("MUX")

    def __str__(self):
        if len(self.args) == 0:
            return self.fun
        if self.fun == "NOT" and len(self.args[0].args) == 0:
            return "!" + self.args[0].fun
        return self.fun + "(" + ",".join([str(a) for a in self.args]) + ")"

    def as_genlib_term(self):
        if len(self.args) == 0:
            return self.fun
        if self.fun == "NOT":
            assert len(self.args[0].args) == 0
            return "!" + self.args[0].fun
        if self.fun == "AND":
            return "(" + self.args[0].as_genlib_term() + "*" + self.args[1].as_genlib_term() + ")"
        if self.fun == "OR":
            return "(" + self.args[0].as_genlib_term() + "+" + self.args[1].as_genlib_term() + ")"
        assert False

    def mapMux(self):
        if self.fun == "MUX":
            A, B, C = self.args
            return OR(AND(A, NOT(C)), AND(B, C)).mapMux()
        return FNode(self.fun, *[a.mapMux() for a in self.args])

    def mapXor(self):
        if self.fun == "XOR":
            A, B = self.args
            return OR(AND(A, NOT(B)), AND(NOT(A), B)).mapXor()
        return FNode(self.fun, *[a.mapXor() for a in self.args])

    def mapNot(self):
        if self.fun == "BUF":
            return self.arg1.mapNot()
        if self.fun == "NOT":
            if self.args[0].fun == "AND":
                return OR(NOT(self.args[0].args[0]),NOT(self.args[0].args[1])).mapNot()
            if self.args[0].fun == "OR":
                return AND(NOT(self.args[0].args[0]),NOT(self.args[0].args[1])).mapNot()
            if self.args[0].fun == "NOT":
                return self.args[0].args[0].mapNot()
        return FNode(self.fun, *[a.mapNot() for a in self.args])

    def map(self):
        n = self
        n = n.mapMux()
        n = n.mapXor()
        n = n.mapNot()
        return n

    def isInv(self):
        if len(self.args) == 0:
            return False
        if self.fun == "XOR":
            return False
        if self.fun == "NOT":
            return self.args[0].isNonInv()
        for a in self.args:
            if not a.isInv():
                return False
        return True

    def isNonInv(self):
        if len(self.args) == 0:
            return True
        if self.fun == "XOR":
            return False
        if self.fun == "NOT":
            return self.args[0].isInv()
        for a in self.args:
            if not a.isNonInv():
                return False
        return True

A = FNode("A")
B = FNode("B")
C = FNode("C")
D = FNode("D")
E = FNode("E")

def BUF(arg): return FNode("BUF", arg)
def NOT(arg): return FNode("NOT", arg)
def AND(arg1, arg2): return FNode("AND", arg1, arg2)
def  OR(arg1, arg2): return FNode( "OR", arg1, arg2)
def XOR(arg1, arg2): return FNode("XOR", arg1, arg2)
def MUX(arg1, arg2, arg3): return FNode("MUX", arg1, arg2, arg3)

# Genlib Format:
#
# GATE <cell-name> <cell-area> <cell-logic-function>
#
# PIN <pin-name> <phase> <input-load> <max-load>
#     <rise-block-delay> <rise-fanout-delay>
#     <fall-block-delay> <fall-fanout-delay>
#
# phase:
#     INV, NONINV, or UNKNOWN
#
# cell-logic-function:
#     <output> = <term with *(AND), +(OR), !(NOT)>

def mkGate(name, cost, expr, max_load=9999, block_delay = 10, fanout_delay = 5):
    expr = expr.map()

    phase = "UNKNOWN"
    if expr.isInv(): phase = "INV"
    if expr.isNonInv(): phase = "NONINV"

    print()
    print("GATE %s %d Y=%s;" % (name, cost, expr.as_genlib_term()))
    print("PIN * %s 1 %d %d %d %d %d" % (phase, max_load, block_delay, fanout_delay, block_delay, fanout_delay))

print("GATE ZERO 0 Y=CONST0;")
print("GATE ONE 0 Y=CONST1;")

mkGate("BUF", 10, A, 2)
mkGate("NOT",  0, NOT(A), 1)

mkGate("AND", 20, AND(A, B), 1)
mkGate( "OR", 20,  OR(A, B), 1)
mkGate("XOR", 20, XOR(A, B), 1)
mkGate("MUX", 20, MUX(A, B, C), 1)

mkGate("AND3", 10, AND(AND(A, B), C), 1)
mkGate(" OR3", 10,  OR( OR(A, B), C), 1)
mkGate("XOR3", 10, XOR(XOR(A, B), C), 1)
mkGate("AOI3", 10, NOT(OR(AND(A, B), C)), 1)
mkGate("OAI3", 10, NOT(AND(OR(A, B), C)), 1)
mkGate("AOI4", 10, NOT(OR(AND(A, B), AND(C, D))), 1)
mkGate("OAI4", 10, NOT(AND(OR(A, B),  OR(C, D))), 1)

for name, expr in [
    ["CC_AAA", AND(AND(A,B),AND(C,D))],
    ["CC_AXA", AND(XOR(A,B),AND(C,D))],
    ["CC_XAX", XOR(AND(A,B),XOR(C,D))],
    ["CC_AAX", AND(AND(A,B),XOR(C,D))],
    ["CC_AXX", XOR(AND(A,B),XOR(C,D))],
    ["CC_XXX", XOR(XOR(A,B),XOR(C,D))],
    ["CC_AAO", AND(AND(A,B), OR(C,D))],
    ["CC_AOA", AND( OR(A,B),AND(C,D))],
    ["CC_AOX", AND( OR(A,B),XOR(C,D))],
]:
    mkGate(name, 5, expr)
    mkGate(name + "_A", 5, AND(expr, E))
    mkGate(name + "_O", 5,  OR(expr, E))
    mkGate(name + "_X", 5, XOR(expr, E))
