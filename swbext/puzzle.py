#!/usr/bin/env python

W = 13
rounds = 1
apos = [0]
bpos = [8]
cpos = [16]

print(f"""
(set-logic ALL)

(declare-fun apos () (_ BitVec 32))
(declare-fun bpos () (_ BitVec 32))
(declare-fun cpos () (_ BitVec 32))
(declare-fun magic ((_ BitVec {W})) (_ BitVec 8))

(assert (bvult apos bpos))
(assert (bvult bpos cpos))

(define-fun f ((h (_ BitVec 32)) (z (_ BitVec 8))) (_ BitVec 8)
    (bvand (bvxor
        (magic ((_ extract {W-1} 0) (bvlshr h apos)))
        (magic ((_ extract {W-1} 0) (bvlshr h bpos)))
        (magic ((_ extract {W-1} 0) (bvlshr h cpos)))
    ) z)
)
""")

for n, v in [("apos", apos), ("bpos", bpos), ("cpos", cpos)]:
    print(f"(assert (or")
    for i in v:
        print(f"  (= {n} #x{i:08x})")
    print(f"))")

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

        h = (v << 8) | m
        for i in range(rounds):
            h = xorshift32(h)

        print(f"(assert (= (f #b{h:032b} #x{z:02x}) #x{y:02x})) ;  rule {index:4}:  {v:08b} {m:08b} -> {y:08b}")

print("(check-sat)")
print("(get-value (")
for i in range(1 << W):
    print(f"  (magic #b{i:0{W}b})")
print("))")
