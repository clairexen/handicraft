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


# Arithmetic functions for binary polynomials

def mul_poly(a, b):
    x = 0
    while b != 0:
        i = b.bit_length()-1
        x ^= a << i
        b ^= 1 << i
    return x

def clmul(a, b):
    return mul_poly(a, b) & 0xffffffff

def clmulh(a, b):
    return (mul_poly(a, b) >> 32) & 0xffffffff


# Xorshift32 implementations

def xorshift32(x):
    if False:
        x = (x ^ (x << 13)) & 0xffffffff
        x = (x ^ (x >> 17)) & 0xffffffff
        x = (x ^ (x <<  5)) & 0xffffffff
    else:
        x = clmul(x, 0x00002001)
        x = (x ^ (x >> 17)) & 0xffffffff
        x = clmul(x, 0x00000021)
    return x

def xorshift32_inv_shl13(x):
    if False:
        t = x ^ (x << 13) & 0xffffffff
        t = x ^ (t << 13) & 0xffffffff
    else:
        t = clmul(x, 0x04002001)
    return t

def xorshift32_inv_shr17(x):
    t = x ^ (x >> 17) & 0xffffffff
    return t

def xorshift32_inv_shl5(x):
    if False:
        t = x ^ (x << 5) & 0xffffffff
        t = x ^ (t << 5) & 0xffffffff
        t = x ^ (t << 5) & 0xffffffff
        t = x ^ (t << 5) & 0xffffffff
        t = x ^ (t << 5) & 0xffffffff
        t = x ^ (t << 5) & 0xffffffff
    else:
        t = clmul(x, 0x42108421)
    return t

def xorshift32_inv(x):
    x = xorshift32_inv_shl5(x)
    x = xorshift32_inv_shr17(x)
    x = xorshift32_inv_shl13(x)
    return x


# Test

n = 123456789
for i in range(10):
    print(n)
    next_n = xorshift32(n)
    assert n == xorshift32_inv(next_n)
    n = next_n
