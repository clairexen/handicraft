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
    name = name.replace(" ", "")
    expr = expr.map()

    phase = "UNKNOWN"
    if expr.isInv(): phase = "INV"
    if expr.isNonInv(): phase = "NONINV"

    print()
    print("GATE %s %d Y=%s;" % (name, cost, expr.as_genlib_term()))
    print("PIN * %s 1 %d %d %d %d %d" % (phase, max_load, block_delay, fanout_delay, block_delay, fanout_delay))

print("GATE ZERO 0 Y=CONST0;")
print("GATE ONE 0 Y=CONST1;")

mkGate("CC_BUF", 5, A)
mkGate("CC_NOT", 0, NOT(A))
mkGate("CC_MUX", 5, MUX(A, B, C))

base_cells = [
    ["CC2_A", AND(A, B)],
    ["CC2_O",  OR(A, B)],
    ["CC2_X", XOR(A, B)],

    ["CC3_AA", AND(AND(A, B), C)],
    ["CC3_OO",  OR( OR(A, B), C)],
    ["CC3_XX", XOR(XOR(A, B), C)],
    ["CC3_AO",  OR(AND(A, B), C)],
    ["CC3_OA", AND( OR(A, B), C)],
    ["CC3_AX", XOR(AND(A, B), C)],
    ["CC3_XA", AND(XOR(A, B), C)],

#   ["CC3_AAA", AND(AND(A,B),AND(A,C))],
#   ["CC3_AXA", XOR(AND(A,B),AND(A,C))],
#   ["CC3_XAX", AND(XOR(A,B),XOR(A,C))],
#   ["CC3_AAX", AND(AND(A,B),XOR(A,C))],
#   ["CC3_AXX", XOR(AND(A,B),XOR(A,C))],
#   ["CC3_XXX", XOR(XOR(A,B),XOR(A,C))],
#   ["CC3_AAO", AND(AND(A,B), OR(A,C))],
#   ["CC3_AOA",  OR(AND(A,B),AND(A,C))],
#   ["CC3_AOX",  OR(AND(A,B),XOR(A,C))],

#   ["CC3_AAA_N", AND(AND(A,B),AND(NOT(A),C))],
#   ["CC3_AXA_N", XOR(AND(A,B),AND(NOT(A),C))],
#   ["CC3_XAX_N", AND(XOR(A,B),XOR(NOT(A),C))],
#   ["CC3_AAX_N", AND(AND(A,B),XOR(NOT(A),C))],
#   ["CC3_AXX_N", XOR(AND(A,B),XOR(NOT(A),C))],
#   ["CC3_XXX_N", XOR(XOR(A,B),XOR(NOT(A),C))],
#   ["CC3_AAO_N", AND(AND(A,B), OR(NOT(A),C))],
#   ["CC3_AOA_N",  OR(AND(A,B),AND(NOT(A),C))],
#   ["CC3_AOX_N",  OR(AND(A,B),XOR(NOT(A),C))],

    ["CC4_AAA", AND(AND(A,B),AND(C,D))],
    ["CC4_AXA", XOR(AND(A,B),AND(C,D))],
    ["CC4_XAX", AND(XOR(A,B),XOR(C,D))],
    ["CC4_AAX", AND(AND(A,B),XOR(C,D))],
    ["CC4_AXX", XOR(AND(A,B),XOR(C,D))],
    ["CC4_XXX", XOR(XOR(A,B),XOR(C,D))],
    ["CC4_AAO", AND(AND(A,B), OR(C,D))],
    ["CC4_AOA",  OR(AND(A,B),AND(C,D))],
    ["CC4_AOX",  OR(AND(A,B),XOR(C,D))],
]

for name, expr in base_cells:
    mkGate(name, 10, expr)

    name = (name
        .replace("CC4_", "CC5_")
        .replace("CC3_", "CC4_")
        .replace("CC2_", "CC3_"))

    mkGate(name + "_A", 12, AND(expr, E))
    mkGate(name + "_O", 12,  OR(expr, E))
    mkGate(name + "_X", 12, XOR(expr, E))
