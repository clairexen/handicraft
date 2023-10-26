#!/usr/bin/env python

W = 13
wmsk = (1 << W)-1

print(f"""
(set-logic ALL)

(declare-fun magic ((_ BitVec {W})) (_ BitVec 8))

(define-fun f ((a (_ BitVec {W})) (b (_ BitVec {W})) (c (_ BitVec {W})) (z (_ BitVec 8))) (_ BitVec 8)
    (bvand (bvxor (magic a) (magic b) (magic c)) z)
)
""")

def bext(value, mask):
    v, m = bin(value)[2:], bin(mask)[2:]
    return int("0"+"".join([a if b == "1" else "" for a, b in zip(v, m)]), 2)

def xorshift32(x):
    x = (x ^ (x << 13)) & 0xFFFFFFFF
    x = (x ^ (x >> 17)) & 0xFFFFFFFF
    x = (x ^ (x <<  5)) & 0xFFFFFFFF
    return x

index = 1
for v in range(256):
    for m in range(256):
        if (v & ~m) != 0: continue
        index += 1
        y = bext(v, m)
        z = bext(m, m)

        x = (v << 8) | m
        x = xorshift32(x)
        a = (x >>  0) & wmsk
        b = (x >>  8) & wmsk
        c = (x >> 16) & wmsk

        print(f"(assert (= (f #b{a:0{W}b} #b{b:0{W}b} #b{c:0{W}b} #x{z:02x}) #x{y:02x})) ;  rule {index:4}:  {v:08b} {m:08b} -> {y:08b}")

print("(check-sat)")
print("(get-value (")
for i in range(1 << W):
    print(f"  (magic #b{i:0{W}b})")
print("))")
