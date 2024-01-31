#!/usr/bin/env python3

import gzip
import numpy as np
from scipy.optimize import linprog

name_to_idx = dict()
idx_to_name = dict()
num_names = 0

A = list()
B = list()

with gzip.open("timing3.txt.gz", "rt") as f:
    line_nr = 0

    for line in f:
        line_nr += 1

        if line_nr == 200:
            break

        net, src_bel, dst_bel, ico, fast_max, fast_min, slow_max, slow_min, pips, inodes, wires = line.split()
        if net == "net": continue

        terms = dict()

        def add_term(name):
            global name_to_idx, idx_to_name, num_names

            if name not in name_to_idx:
                name_to_idx[name] = num_names
                idx_to_name[num_names] = name
                num_names += 1

            idx = name_to_idx[name]

            if idx not in terms:
                terms[idx] = 1
            else:
                terms[idx] += 1

        for pip in pips.split("|"):
            _, name = pip.split(":")
            add_term("p" + name)


        for wire in wires.split("|"):
            _, name = wire.split(":")
            add_term("w" + name)

        A.append(terms)
        B.append([float(fast_max), float(fast_min), float(slow_max), float(slow_min)])

print("Total number of delays: %d" % num_names)
print("Total number of observations: %d" % len(A))

for i in range(len(A)):
    row = list()
    for j in range(num_names):
        if j in A[i]:
            row.append(A[i][j])
        else:
            row.append(0)
    A[i] = row

A = np.array(A, np.float64)
B = np.array(B, np.float64)

print("Numpy A array shape: %s" % (A.shape,))
print("Numpy B array shape: %s" % (B.shape,))

option_lstsq = False
option_rndsum = False

if option_lstsq:
    x, _, _, _ = np.linalg.lstsq(A, B)
    for i in range(num_names):
        print("%-10s %8.2f %8.2f %8.2f %8.2f" % (idx_to_name[i], x[i, 0], x[i, 1], x[i, 2], x[i, 3]))

else:
    c = np.ones(num_names)
    if option_rndsum:
        A_ub = -A
        b_ub = -B[:,0]
        for i in range(10):
            k = np.random.randint(A_ub.shape[0])
            p = np.random.uniform()
            A_ub += p * np.roll(A_ub, k, 0)
            b_ub += p * np.roll(b_ub, k, 0)
    else:
        A_ub = -A
        b_ub = -B[:,0]
    lpres = linprog(c, A_ub, b_ub)
    print(lpres)
    for i in range(num_names):
        print("%-10s %8.2f" % (idx_to_name[i], lpres.x[i]))


