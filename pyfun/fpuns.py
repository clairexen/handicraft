#!/usr/bin/env python3
#
# Experiments with floating point representations, comparing:
#   - Fpuns (newly proposed format by Claire Xenia Wolf)
#   - Posits (aka type-III Unums, as proposed by John L. Gustafson)
#   - IEEE floating point numbers

import math
import numpy as np

def bvNeg(a):
    a = ~np.array(a, np.bool_)
    for i in range(len(a)):
        a[i] = not a[i]
        if a[i]: break
    return a

def bvAbs(a):
    if a[-1]:
        return bvNeg(a)
    return a

class Fpun:
    def __init__(self, bits, size=None):
        if size is None:
            self.data = np.array(bits, np.bool_)
        else:
            self.data = np.array([((bits>>i)&1) == 1 for i in range(size)], np.bool_)

        negative = self.data[-1]
        superNormal = self.data[-2] != self.data[-1]
        bits = bvAbs(self.data)

        exponent = 0
        cursor = 3
        ebits = 0
        overflow = 0

        while cursor <= len(bits):
            if bits[-cursor] != bits[-2]:
                cursor += 1
                break
            ebits += exponent
            exponent = 1
            cursor += 1
        else:
            if not bits[-2]:
                self.nonzero = False
                self.value = None if self.data[-1] else 0
                return

        for i in range(ebits):
            exponent <<= 1
            if cursor <= len(bits):
                if bits[-cursor] == superNormal:
                    exponent |= 1
                cursor += 1
            else:
                if not superNormal:
                    exponent |= 1
                overflow += 1

        if not superNormal:
            exponent = -1-exponent

        mantissa = 1
        while cursor <= len(bits):
            mantissa <<= 1
            if bits[-cursor]:
                mantissa |= 1
            cursor += 1

        if negative:
            mantissa = -mantissa

        self.exponent = exponent
        self.mantissa = mantissa
        self.expbits = ebits - overflow
        self.overflow = overflow
        self.mantbits = mantissa.bit_length()-1

        self.nonzero = True
        self.value = None if abs(exponent) > 100 else \
                math.pow(2, exponent) * mantissa / (1 << (mantissa.bit_length()-1))

    def describe(self):
        s = []
        bits = list(["1" if bit else "0" for bit in bvAbs(self.data)])

        if not self.nonzero:
            bits = ["="] + bits + ["="]
        else:
            bits.insert(self.mantbits + self.expbits, ":")
            bits.insert(self.mantbits, ">" if self.overflow else "|")

        s.append("".join(reversed(bits)) + "  =>  ")

        if not self.nonzero:
            if self.value is None:
                s.append("NaN")
            else:
                s.append("Zero")
        else:
            s.append("-" if self.mantissa < 0 else "+")
            s.append(f"exp2({self.exponent:+3d}) * ")
            s.append(bin(abs(self.mantissa)).replace("0b1", "1.").ljust(7, "_") + " (bin)")

            if abs(self.value) > 256:
                s.append(f" = {int(self.value):12d}")
            else:
                s.append(f" = {self.value:12.5f}")

            if abs(1/self.value) > 256:
                s.append(f"  (= {int(1/self.value):12d} ^-1)")
            else:
                s.append(f"  (= {1/self.value:12.3f} ^-1)")

        return "".join(s)

def genPositiveValues(nbits=8):
    return list([val for val in [
        Fpun(i, nbits).value for i in range(1, 1 << (nbits-1))
    ] if val is not None])

if __name__ == "__main__":
    for i in range(-128, 128):
        p = Fpun(i, 8)
        print(f"{i:4d} {i&255:08b}:  {p.describe()}")
    print(genFpunValues())
