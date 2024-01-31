#!/usr/bin/env python3
#
# https://twitter.com/robeastaway/status/1255558573712707585

from pysmt.shortcuts import Symbol, Int, Ite, And, Or, LE, Equals, NotEquals, get_model
from pysmt.typing import INT

#%%

N = 4   # type of polygon
M = 12  # max value

squares = list()
constraints = list()

def add_square():
    i = len(squares)
    squares.append([Symbol("sym_%d_%d" % (i, k), INT) for k in range(N)])
    constraints.append(Or(*[NotEquals(squares[i][k], Int(0)) for k in range(N)]))
    if i > 0:
        for k in range(N):
            constraints.append(Equals(squares[i][k], Ite(squares[i-1][k] > squares[i-1][(k+1)%N],
                               squares[i-1][k] - squares[i-1][(k+1)%N], squares[i-1][(k+1)%N] - squares[i-1][k])))
    else:
        for k in range(N):
            constraints.append(And(LE(Int(1), squares[i][k]), LE(squares[i][k], Int(M))))

while True:
    add_square()
    model = get_model(And(constraints), solver_name="yices")
    if model:
        print(len(squares))
        for i in range(len(squares)):
            print(" ", end="")
            for k in range(N):
                print(" %2d" % model.get_value(squares[i][k]).constant_value(), end="")
            print()
    else:
        break
