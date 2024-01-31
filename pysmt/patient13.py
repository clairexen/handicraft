#!/usr/bin/env python3
#
# https://twitter.com/vardi/status/1253683492975775747
#
#   1 -  2 -  3 -  4
#   |    |    |    |
#   5 -  6 -  7 -  8
#   |    |    |    |
#   9 - 10 - 11 - 12
#   |    |    |    |
#  13 - 14 - 15 - 16
#
#
#  Simple reading of the problem statement:
#
#  Find a path from 13 to 4 that visits each node exactly
#  once, or prove that it is not possible.
#
#
#  Proper reading of the problem statement:
#
#  Like above, but room 13 can be re-visited because it
#  is empty. This can be modelled by using node 14 or 9
#  as start node instead of 13.

from pysmt.shortcuts import Symbol, Int, Ite, Iff, And, Equals, get_model
from pysmt.typing import INT, BOOL

#%%

graph = {
    1: set([2, 5]),
    2: set([1, 6, 3]),
    3: set([2, 7, 4]),
    4: set([3, 8]),

    5: set([1, 9, 6]),
    6: set([5, 2, 10, 7]),
    7: set([6, 3, 11, 8]),
    8: set([7, 4, 12]),

    9: set([5, 13, 10]),
    10: set([9, 6, 14, 11]),
    11: set([10, 7, 15, 12]),
    12: set([11, 8, 16]),

    13: set([9, 14]),
    14: set([13, 10, 15]),
    15: set([14, 11, 16]),
    16: set([15, 12])
}

start_node = 14
final_node = 4

idx_symbols = dict([(k, Symbol("idx_%d" % k, INT)) for k in graph.keys()])
edge_symbols = dict([((k,n), Symbol("edge_%d_%d" % (k, n), BOOL)) for k,e in graph.items() for n in e])

constraints = list()
constraints.append(Equals(idx_symbols[start_node], Int(1)))
constraints.append(Equals(idx_symbols[final_node], Int(len(graph))))

for k,e in graph.items():
    # check graph DB is consistent
    for n in e:
        assert k in graph[n]

    # each node must have exactly one active in-edge, except the start node
    s = Int(0)
    for n in e:
        s += Ite(edge_symbols[n,k], Int(1), Int(0))
    if k != start_node:
        constraints.append(Equals(s, Int(1)))
    else:
        constraints.append(Equals(s, Int(0)))

    # each node must have exactly one out-edge, except the final node
    s = Int(0)
    for n in e:
        s += Ite(edge_symbols[k,n], Int(1), Int(0))
    if k != final_node:
        constraints.append(Equals(s, Int(1)))
    else:
        constraints.append(Equals(s, Int(0)))

    # constraint indices
    for n in e:
        constraints.append(Iff(edge_symbols[k,n], Equals(idx_symbols[k] + Int(1), idx_symbols[n])))

#%%

model = get_model(And(constraints))

if model:
    print("SAT")
    d = dict()
    for k in graph.keys():
        v = model.get_value(idx_symbols[k]).constant_value()
        d[v] = k
    print(" ".join(["%d" % d[i] for i in range(1, len(graph)+1)]))
else:
    print("UNSAT")
