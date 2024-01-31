#!/usr/bin/env python3
#
#  Copyright (C) 2017  Clifford Wolf <clifford@clifford.at>
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
#

from sys import argv
from subprocess import Popen, PIPE
import numpy as np

N = 16

if len(argv) > 1:
    N = int(argv[1])

if len(argv) > 2:
    np.random.seed(int(argv[2]))

from_bits = np.arange(32)
to_bits = np.arange(32)

np.random.shuffle(from_bits)
np.random.shuffle(to_bits)

from_bits = from_bits[0:N]
to_bits = to_bits[0:N]

pattern = ["-" for i in range(32)]
symbols = "0123456789abcdefghijklmnopqrstuv"

for i in range(N):
    pattern[31-from_bits[i]] = symbols[to_bits[i]]

pattern = "".join(pattern)

print("// %s" % from_bits)
print("// %s" % to_bits)
print("// %s" % pattern)

print("""
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

uint32_t rot(uint32_t v, int k)
{
  return k ? (v << (32-k)) | (v >> k) : 0;
}

uint32_t grev(uint32_t x, int k)
{
	if (k & 1 ) x = ((x & 0x55555555) << 1 ) | ((x & 0xAAAAAAAA) >> 1 );
	if (k & 2 ) x = ((x & 0x33333333) << 2 ) | ((x & 0xCCCCCCCC) >> 2 );
	if (k & 4 ) x = ((x & 0x0F0F0F0F) << 4 ) | ((x & 0xF0F0F0F0) >> 4 );
	if (k & 8 ) x = ((x & 0x00FF00FF) << 8 ) | ((x & 0xFF00FF00) >> 8 );
	if (k & 16) x = ((x & 0x0000FFFF) << 16) | ((x & 0xFFFF0000) >> 16);
	return x;
}

uint32_t bext(uint32_t v, uint32_t mask)
{
  int t = 0;
  uint32_t c = 0;
  for (int i = 0; i < 32; i++) {
    if (mask & 1) {
      c |= (v & 1) << (t++);
    }
    v >>= 1;
    mask >>= 1;
  }
  return c;
}

uint32_t bdep(uint32_t v, uint32_t mask)
{
  uint32_t c = 0;
  for (int i = 0; i < 32; i++) {
    if (mask & 1) {
      c |= (v & 1) << i;
      v >>= 1;
    }
    mask >>= 1;
  }
  return c;
}

uint32_t and(uint32_t v1, uint32_t v2)
{
  return v1 & v2;
}

uint32_t or(uint32_t v1, uint32_t v2)
{
  return v1 | v2;
}
""")

print("uint32_t perm(uint32_t v0)")
print("{")
with Popen(["./permsyn", "-c", pattern], stdout=PIPE) as proc:
    for line in proc.stdout.readlines():
        print("  " + line.decode("ascii"), end="")
print("}")
print()

print("void check(uint32_t v, uint32_t ref)")
print("{")
print("  uint32_t out = perm(v);")
print("  printf(\"testing 0x%08x -> 0x%08x: 0x%08x %s\\n\", v, ref, out, ref == out ? \"OK\" : \"ERROR\");")
print("  if (ref != out) exit(1);")
print("}")
print()

print("int main()")
print("{")

for k in range(100):
    vin = np.random.randint(0, 0x100000000)
    vout = 0
    for i in range(N):
        if (vin & (1 << from_bits[i])) != 0:
            vout |= 1 << to_bits[i]
    print("  check(0x%08x, 0x%08x);" % (vin, vout))

print("  return 0;")
print("}")

