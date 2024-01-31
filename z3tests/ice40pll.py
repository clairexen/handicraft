#!/usr/bin/env python

from z3 import *

s = Solver()
FEEDBACK_PATH = "simple"

# Frequencies in MHz
F_PLLOUT = Real('F_PLLOUT')
F_PLLIN = Real('F_PLLIN')

DIVF = Int('DIVF')
DIVR = Int('DIVR')
DIVQ = Int('DIVQ')

s.add(DIVF >= 0)
s.add(DIVF <= 63)

s.add(DIVR >= 0)
s.add(DIVR <= 15)

s.add(DIVQ >= 1)
s.add(DIVQ <= 6)

s.add(F_PLLIN >=  10)
s.add(F_PLLIN <= 133)

s.add(F_PLLOUT >=  16)
s.add(F_PLLOUT <= 275)

F_VCO = Real('F_VCO')
s.add(F_VCO * DIVR == F_PLLIN * DIVF)
s.add(F_VCO >=  533)
s.add(F_VCO <= 1066)

F_PFD = Real('F_PFD')
s.add(F_PFD * DIVR == F_PLLIN)
s.add(F_PFD >=  10)
s.add(F_PFD <= 133)

if FEEDBACK_PATH != "SIMPLE":
    s.add(F_PLLOUT * (DIVR+1) == F_PLLIN * (DIVF+1))
else:
    s.add(F_PLLOUT * (2 ** DIVQ) * (DIVR+1) == F_PLLIN * (DIVF+1))

s.add(F_PLLIN == 12)
s.add(F_PLLOUT > 50)
s.add(F_PLLOUT < 60)

print(s.check())
print(s.model())

