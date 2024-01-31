#!/usr/bin/env python3

minstret_state = 0

def nop_insn_model():
    global minstret_state
    minstret_state += 1

# This is what spike seems to do
def csrrc_insn_model_A(arg):
    global minstret_state
    r = minstret_state
    minstret_state = minstret_state & ~arg
    minstret_state += 1
    return r

def csrrc_insn_model_B(arg):
    global minstret_state
    r = minstret_state
    minstret_state += 1
    minstret_state = minstret_state & ~arg
    return r

def csrrc_insn_model_C(arg):
    global minstret_state
    r = minstret_state
    minstret_state = minstret_state & ~arg
    return r

for name, csrrc_insn_model in [("A", csrrc_insn_model_A), ("B", csrrc_insn_model_B), ("C", csrrc_insn_model_C)]:
    # csrrwi x0, minstret, 0
    csrrc_insn_model(-1)

    # nop
    nop_insn_model()

    # csrrci x0, minstret, 1
    csrrc_insn_model(1)

    # csrrci x1, minstret, 0
    v = csrrc_insn_model(0)

    print("Model %s: %d" % (name, v))
