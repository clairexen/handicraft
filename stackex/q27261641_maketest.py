#!/usr/bin/python

from __future__ import division
from __future__ import print_function
import numpy as np

cnf = list()
maxvar = np.random.randint(10, 20)

for i in range(0, np.random.randint(100, 400)):
    cnf.append(list())
    for j in range(0, np.random.randint(2, 20)):
        cnf[-1].append(np.random.randint(maxvar)+1)
        if np.random.randint(2):
            cnf[-1][-1] *= -1

print("p cnf %d %d" % (maxvar, len(cnf)))
for clause in cnf:
    print(" ".join([ "%d" % lit for lit in clause ]) + " 0")

