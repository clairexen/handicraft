#!/usr/bin/env python3
#
#  Copyright (C) 2018  Clifford Wolf <clifford@clifford.at>
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

print()
print("Calculate coefficients for the CRC folding method described in \"Fast CRC Computation Using PCLMULQDQ Instruction\"")
print("[1] https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/fast-crc-computation-generic-polynomials-pclmulqdq-paper.pdf")

def reflect(m, sz, nblocks=1):
    for i in range(nblocks):
        b = (m >> (sz*i)) & ((1 << sz)-1)

        x = 0
        for j in range(sz):
            if (b >> j) & 1:
                x |= 1 << (sz-j-1)

        m = m ^ ((b ^ x) << (sz*i))
    return m

def mod_div_poly(data, poly):
    quotient = 0
    while data.bit_length() >= poly.bit_length():
        i = data.bit_length() - poly.bit_length()
        data = data ^ (poly << i)
        quotient |= 1 << i
    return (data, quotient)

def mod_poly(data, poly):
    return mod_div_poly(data, poly)[0]

def div_poly(data, poly):
    return mod_div_poly(data, poly)[1]

def str2int(s):
    v = 0
    for c in s:
        v = (v << 8) | ord(c)
    return v

def int2words(v):
    a = []
    while v:
        a.append(v & ((1 << 64)-1))
        v = v >> 64
    return list(reversed(a))

def words2int(a):
    v = 0
    for w in a:
        v = (v << 64) | w
    return v

def clmul(a, b):
    v = 0
    for i in range(b.bit_length()):
        if b & (1 << i):
            v = v ^ (a << i)
    return v

# -------------------------------------------------------------------

def generic_clmul_crc32(m, P, k1, k2, k3, mu):
    print()
    print("-- Reduce message to 2x 64 bit --")

    m = int2words(m)
    print("Message (as 64-bit words): [%s]" % ", ".join(["0x%016X" % v for v in m]))

    while len(m) > 2:
        a = clmul(m[0], k1) >> 64
        b = clmul(m[0], k1) & ((1<<64)-1)
        print("CLMUL: 0x%016x x 0x%016x = 0x%016x_%016x" % (m[0], k1, a, b))

        m[1] ^= a
        m[2] ^= b
        m = m[1:]
        print("Message (as 64-bit words): [%s]" % ", ".join(["0x%016X" % v for v in m]))

    print()
    print("-- Reduce message to 64 bit, add 32 bit zero padding --")

    a = clmul(m[0], k2) >> 64
    b = clmul(m[0], k2) & ((1<<64)-1)
    print("CLMUL: 0x%08x x 0x%08x = 0x%08x_%016x" % (m[0] >> 32, k2, a, b))

    m[0] = (m[1] >> 32) ^ a
    m[1] = ((m[1] << 32) ^ b) & ((1<<64)-1)
    print("Message (as 64-bit words): [%s]" % ", ".join(["0x%016X" % v for v in m]))

    a = clmul(m[0], k3)
    print("CLMUL: 0x%08x x 0x%08x = 0x%016x" % (m[0], k3, a))
    m[1] ^= a
    m = m[1:]

    m = words2int(m)
    print("Message (as large int): 0x%X" % m)

    print()
    print("-- Barrett Reduction --")

    t1 = clmul(m >> 32, mu)
    print("CLMUL: 0x%08x x 0x%08x = 0x%016x" % (m >> 32, mu, t1))
    t2 = clmul(t1 >> 32, P)
    print("CLMUL: 0x%08x x 0x%08x = 0x%016x" % (t1 >> 32, P, t2))
    return m ^ t2

# -------------------------------------------------------------------

# CRC32 Polynomial
P = 0x104c11db7

print()
print("Calculate values from [1, p. 16]:")
print("P   = 0x%09X" % (P))
print("k1  =  0x%08X" % (mod_poly(1 << (4*128+64), P)))
print("k2  =  0x%08X" % (mod_poly(1 << (4*128), P)))
print("k3  =  0x%08X" % (mod_poly(1 << (128+64), P)))
print("k4  =  0x%08X" % (mod_poly(1 << 128, P)))
print("k5  =  0x%08X" % (mod_poly(1 << 96, P)))
print("k6  =  0x%08X" % (mod_poly(1 << 64, P)))
print("mu  = 0x%09X"  % (div_poly(1 << 64, P)))

print()
print("Calculate values from [1, p. 22]:")
print("P'  = 0x%09X" % (reflect(P, 33)))
print("k1' = 0x%09X" % (reflect(mod_poly(1 << (4*128+32), P) << 32, 64) << 1))
print("k2' = 0x%09X" % (reflect(mod_poly(1 << (4*128-32), P) << 32, 64) << 1))
print("k3' = 0x%09X" % (reflect(mod_poly(1 << (128+32), P) << 32, 64) << 1))
print("k4' = 0x%09X" % (reflect(mod_poly(1 << (128-32), P) << 32, 64) << 1))
print("k5' = 0x%09X" % (reflect(mod_poly(1 << 64, P) << 32, 64) << 1))
print("k6' = 0x%09X" % (reflect(mod_poly(1 << 32, P) << 32, 64) << 1))
print("mu' = 0x%09X" % (reflect(div_poly(1 << 64, P), 33)))

# CRC32Q
P = 0x1814141ab

print()
print("The values we need for CRC-32Q:")
print("P   = 0x%09X" % P)
print("k1  =  0x%08X" % (mod_poly(1 << 128, P)))
print("k2  =  0x%08X" % (mod_poly(1 <<  96, P)))
print("k3  =  0x%08X" % (mod_poly(1 <<  64, P)))
print("mu  = 0x%09X"  % (div_poly(1 <<  64, P)))

# CRC32
P = 0x104c11db7

print()
print("The values we need for zlib CRC-32:")
print("P   = 0x%09X" % P)
print("k1  =  0x%08X" % (mod_poly(1 << 128, P)))
print("k2  =  0x%08X" % (mod_poly(1 <<  96, P)))
print("k3  =  0x%08X" % (mod_poly(1 <<  64, P)))
print("mu  = 0x%09X"  % (div_poly(1 <<  64, P)))

print()

# -------------------------------------------------------------------

print("=== CRC32Q DEMO ===")

P  = 0x1814141ab
k1 =  0xA1FA6BEC
k2 =  0x9BE9878F
k3 =  0xB1EFC5F6
mu = 0x1FEFF7F62

m = str2int("The quick brown fox jumps over the lazy dog ABCDEFGHIJKLMNOPQRST")

print("Message (as large int): 0x%X" % m)

c = mod_poly(m << 32, P)
print("CRC32Q: 0x%08X" % c)
assert c == 0xA9DE0134

c = generic_clmul_crc32(m, P, k1, k2, k3, mu)
print("CRC32Q: 0x%08X" % c)
assert c == 0xA9DE0134

print()

# -------------------------------------------------------------------

print("=== ZLIB CRC32 DEMO ===")

P  = 0x104c11db7
k1 =  0xE8A45605
k2 =  0xF200AA66
k3 =  0x490D678D
mu = 0x104D101DF

m = str2int("The quick brown fox jumps over the lazy dog ABCDEFGHIJKLMNOPQRST")
m = reflect(m, 8, 64)
m ^= 0xffffffff << (60*8)

print("Message (as large int): 0x%X" % m)

c = mod_poly(m << 32, P)
c = reflect(c, 32) ^ 0xffffffff
print("CRC32: 0x%08X" % c)
assert c == 0x1BE7DE66

c = generic_clmul_crc32(m, P, k1, k2, k3, mu)
c = reflect(c, 32) ^ 0xffffffff

print("CRC32: 0x%08X" % c)
assert c == 0x1BE7DE66

print()
