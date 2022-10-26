#!/usr/bin/env python3

import random
import sys

enableDebug = True

def debug(*args):
    if enableDebug: print(*args)

class BitMaskSet:
    def __init__(self, N, depth, value=None, default=0):
        self.N = N
        self.depth = depth
        self.numRows = 1 << depth
        self.numCols = N >> depth
        self.data = [[default]*self.numCols for i in range(self.numRows)]
        if value is not None: self.set(value)

    def set(self, value):
        if type(value) is str:
            index = 0
            for i in reversed(range(self.N)):
                if value[index] in ("0", "1"):
                    self[i] = int(value[index])
                else:
                    self[i] = value[index]
                index += 1
                if i != 0 and i % self.numCols == 0 and value[index] == " ":
                    index += 1
            assert index == len(value)
            return

        if type(value) is BitMaskSet:
            assert self.N == value.N
            for i in range(self.N):
                self[i] = value[i]
            return

        assert False

    def __str__(self):
        s = list()
        for i in reversed(range(self.numRows)):
            for j in reversed(range(self.numCols)):
                s.append(str(self[i,j]))
            # if i != 0: s.append(" ")
        return "".join(s)

    def __getitem__(self, key):
        if type(key) is tuple:
            return self.data[key[0]][key[1]]
        return self.data[key // self.numCols][key % self.numCols]

    def __setitem__(self, key, value):
        if type(key) is tuple:
            self.data[key[0]][key[1]] = value
        else:
            self.data[key // self.numCols][key % self.numCols] = value

    def xorsum(self, cin=0):
        result = BitMaskSet(self.N>>1, self.depth)
        for i in range(self.numRows):
            carry = cin
            for j in range(self.numCols>>1):
                carry = carry ^ self[i,2*j]
                result[i,j] = carry
                carry = carry ^ self[i,2*j+1]
        return result

    def split(self, mask=None):
        result = BitMaskSet(self.N, self.depth+1)
        for i in range(self.numRows):
            for j in range(self.numCols>>1):
                pair = self[i,2*j], self[i,2*j+1]
                if mask is not None and mask[i,j]:
                    pair = pair[1], pair[0]
                result[2*i,j] = pair[0]
                result[2*i+1,j] = pair[1]
        return result

    def merge(self, mask=None):
        result = BitMaskSet(self.N, self.depth-1)
        for i in range(self.numRows>>1):
            for j in range(self.numCols):
                pair = self[2*i,j], self[2*i+1,j]
                if mask is not None and mask[i,j]:
                    pair = pair[1], pair[0]
                result[i,2*j] = pair[0]
                result[i,2*j+1] = pair[1]
        return result

    def mask(self, other, default=0):
        result = BitMaskSet(self.N, self.depth, default=default)
        for i in range(self.numRows):
            for j in range(self.numCols):
                if self[i,j]: result[i,j] = other[i,j]
        return result

    def inverse(self):
        result = BitMaskSet(self.N, self.depth)
        for i in range(self.numRows):
            for j in range(self.numCols):
                result[i,j] = 1-self[i,j]
        return result

class SAG4Fun:
    def __init__(self, N):
        self.N = N
        self.xcfg = [BitMaskSet(N, i) for i in range((N-1).bit_length())]
        self.extMask = BitMaskSet(N, 0)
        self.depMask = BitMaskSet(N, 0)

    def loadMask(self, value):
        M = BitMaskSet(self.N, 0, value)
        self.extMask = M

        for i in range((self.N-1).bit_length()):
            self.xcfg[i] = M.xorsum(1)
            debug(f"-x{i}", self.xcfg[i])
            M = M.split(self.xcfg[i])

        for i in range((self.N-1).bit_length()):
            M = M.merge()
        self.depMask = M

    def SAG(self, value):
        D = BitMaskSet(self.N, 0, value)

        for i in range((self.N-1).bit_length()):
            D = D.split(self.xcfg[i])
            debug(f"-d{i}", D)

        for i in range((self.N-1).bit_length()):
            D = D.merge()

        return D

    def GAS(self, value):
        D = BitMaskSet(self.N, 0, value)

        for i in range((self.N-1).bit_length()):
            D = D.split()

        for i in reversed(range((self.N-1).bit_length())):
            debug(f"-d{i}", D)
            D = D.merge(self.xcfg[i])

        return D

    def EXT(self, value):
        D = BitMaskSet(self.N, 0, value)
        D = self.extMask.mask(D, "_")
        return self.SAG(D)

    def DEP(self, value):
        D = BitMaskSet(self.N, 0, value)
        D = self.depMask.mask(D, "_")
        return self.GAS(D)

def demo():
    global enableDebug
    enableDebug = True

    letters = "abcdefgh"
    numbers = "12345678"

    sag = SAG4Fun(8)

    for M in ["00110100", "01010110"]:
        print("M: ", M)
        sag.loadMask(M)

        I = ""
        l = letters[0:M.count("0")]
        n = numbers[0:M.count("1")]
        for i in range(8):
            if M[i] == "0":
                I += l[-1]
                l = l[0:-1]
            else:
                I += n[0]
                n = n[1:]
        assert l == "" and n == ""

        print()
        print("D: ", D := I)
        print("--")
        print("ext", sag.EXT(D))
        print("--")
        print("sag", D := sag.SAG(D))

        assert str(D) == "".join(sorted(str(D), key=lambda k: ord(k)%ord("a")))
        print()

        print("D: ", D)
        print("--")
        print("dep", sag.DEP(D))
        print("--")
        print("gas", D := sag.GAS(D))

        assert str(I) == str(D)
        print()
        print("=" * 50)
        print()

def tests(N):
    global enableDebug
    enableDebug = False

    symbols = "#+0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    assert N <= len(symbols)
    symbols = symbols[0:N]

    sag = SAG4Fun(N)

    for i in range(100):
        M = "".join(random.choices("01", k=N))
        sag.loadMask(M)

        I = ""
        l = symbols[:M.count("0")]
        r = symbols[M.count("0"):]
        for i in range(N):
            if M[i] == "0":
                I += l[-1]
                l = l[:-1]
            else:
                I += r[0]
                r = r[1:]
        assert l == "" and r == ""

        D = sag.SAG(I)
        O = sag.GAS(D)

        sep = " " if N<=32 else "\n\t"
        print(f"M={M} I={I}{sep}D={D} O={O}")

        assert str(D) == "".join(sorted(str(D)))
        assert str(I) == str(O)

    print()
    print("=" * 50)
    print()

def checks(N):
    global enableDebug
    enableDebug = False

    symbols = "#+0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    assert N <= len(symbols)
    symbols = symbols[0:N]
    
    sag = SAG4Fun(N)
    log2N = (N-1).bit_length()

    print()
    print("I ", symbols)

    D = BitMaskSet(N, 0, symbols)
    for i in range(log2N):
        D = D.split()
        if i == 0: print(f"S{i}", S := D)
    print(f"S{i}", D)

    D = BitMaskSet(N, log2N, symbols)
    for i in range(log2N):
        D = D.merge()
        if i == 0: print(f"M{i}", M := D)
    print(f"M{i}", D)

    def makeFun(name, value):
        expr = ", ".join([f"in[{symbols.find(c)}]" for c in value])
        print()
        print(f"function [{N-1}:0] {name};")
        print(f"  input [{N-1}:0] in;")
        print(f"  bitrev{N} = {{{expr}}};")
        print(f"endfunction")

    makeFun(f"split{N}", str(S))
    makeFun(f"merge{N}", str(M))
    makeFun(f"bitrev{N}", str(D))

    print()
    print("=" * 50)
    print()

if __name__ == "__main__":
    demo()
    tests(32)
    tests(64)
    checks(32)
    checks(64)
    print("ALL TESTS PASSED")
    print()
    sys.exit(0)
