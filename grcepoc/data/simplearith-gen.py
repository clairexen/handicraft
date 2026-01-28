#!/usr/bin/env python3

import random

def random_expr(max_depth=3, first=True):
    if max_depth <= 0 or (not first and random.random() < 0.35):
        sign = random.choice(["", "+", "-"])
        return f"{sign}{random.randint(0, 99)}"

    if random.random() < 0.1:
        sign = random.choice(["", "+", "-"])
        expr = random_expr(max_depth - 1, first)
        return f"{sign}({expr})"

    left = random_expr(max_depth - 1, False)
    right = random_expr(max_depth - 1, False)
    op = random.choice(["+", "-", "*"])

    expr = f"{left} {op} {right}"

    if random.random() < 0.5:
        expr = f"({expr})"

    return expr

for i in range(10000000):
    print()
    expr = random_expr()
    print(f"{expr} = ?")
    value = eval(expr)
    for j in range(random.randint(-1, 7), -1, -1):
        print(f"<|p{j}|>", end="")
    print(f"\n{value}\n\n<|----|>")
