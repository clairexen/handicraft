#!/usr/bin/env python3

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
            for i in reversed(range(self.numRows)):
                for j in reversed(range(self.numCols)):
                    if value[index] in ("0", "1"):
                        self[i,j] = int(value[index])
                    else:
                        self[i,j] = value[index]
                    index += 1
                if i != 0:
                    assert value[index] == " "
                    index += 1
            assert index == len(value)
            return
        assert False

    def __str__(self):
        s = list()
        for i in reversed(range(self.numRows)):
            for j in reversed(range(self.numCols)):
                s.append(str(self[i,j]))
            if i != 0:
                s.append(" ")
        return "".join(s)

    def __getitem__(self, key):
        return self.data[key[0]][key[1]]

    def __setitem__(self, key, value):
        self.data[key[0]][key[1]] = value

    def xorsum(self, cin=0):
        result = BitMaskSet(self.N, self.depth)
        for i in range(self.numRows):
            carry = cin
            for j in range(self.numCols):
                carry = carry ^ self[i,j]
                result[i,j] = carry
        return result

    def split(self, mask, hi=False):
        result = BitMaskSet(self.N, self.depth+1)
        for i in range(self.numRows):
            for j in range(0, self.numCols, 2):
                pair = self[i,j], self[i,j+1]
                if mask[i, j+1 if hi else j]:
                    pair = pair[1], pair[0]
                result[2*i,j>>1] = pair[0]
                result[2*i+1,j>>1] = pair[1]
        return result

    def merge(self):
        result = BitMaskSet(self.N, self.depth-1)
        for i in range(0, self.numRows, 2):
            for j in range(self.numCols):
                result[i>>1,2*j] = self[i,j]
                result[i>>1,2*j+1] = self[i+1,j]
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
        self.mask = BitMaskSet(N, 0)

    def printState(self):
        for i in range((self.N-1).bit_length()):
            print(f"X{i}:", self.xcfg[i])

    def loadMask(self, value):
        M = BitMaskSet(self.N, 0, value)
        self.mask = M

        for i in range((self.N-1).bit_length()):
            self.xcfg[i] = M.xorsum(1)
            M = M.split(self.xcfg[i])

    def bitExtract(self, value):
        D = BitMaskSet(self.N, 0, "deABfCgh")
        D = self.mask.mask(D, "_")
        for i in range((self.N-1).bit_length()):
            D = D.split(self.xcfg[i])
        for i in range((self.N-1).bit_length()):
            D = D.merge()
        return D

    def bitDeposit(self, value):
        pass  # TBD

def main():
    print(D := "deABfCgh")
    print(M := "00110100")
    sag = SAG4Fun(8)
    sag.loadMask(M)
    # sag.printState()
    print(sag.bitExtract(D))

main()
