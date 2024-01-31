#!/usr/bin/env python3
# Generate SMT2 code for synthesizing RISC-V XBitmanip programs for bit manipulations

import sys, os, re, time, tempfile

class Syn:
    def __init__(self, xlen, perm, tempdir=None, order_mode=False, verbose=True, tagtype="normal", dummy=False):
        self.xlen = xlen
        self.perm = perm
        self.order_mode = order_mode
        self.verbose = verbose
        self.tagtype = tagtype
        self.dummy = dummy
        self.sat = False

        assert self.xlen in [32, 64]
        assert self.tagtype in ["normal", "dense", "onehot"]

        self.log2_xlen = 5 if self.xlen == 32 else 6

        if tempdir is None:
            self.tempdir_obj = tempfile.TemporaryDirectory()
            self.tempdir = self.tempdir_obj.name
        else:
            self.tempdir = tempdir

        if self.verbose:
            print("--src--  --dst--")
            for (p, q), (x, y) in zip(sorted(perm, key=lambda x: x[0]), sorted(perm, key=lambda x: x[1])):
                print(" %2d %2d    %2d %2d" % (p, q, x, y))
            print("Tempdir: %s" % tempdir)

        self.smtfile = open("%s/progsynth.smt2" % self.tempdir, "w")
        print("(set-option :produce-models true)", file=self.smtfile)
        print("(set-logic QF_BV)", file=self.smtfile)

        if self.order_mode:
            self.perm = sorted(self.perm, key=lambda x: x[1])
            assert self.tagtype != "dense"

        if self.tagtype == "onehot":
            self.tagsz = len(perm)

        elif self.tagtype == "dense":
            self.tagsz = self.log2_xlen

        elif self.tagtype == "normal":
            self.tagsz = 1
            while (1 << self.tagsz) <= len(self.perm):
                self.tagsz += 1

        else:
            assert 0

        if self.tagtype == "dense":
            self.cursor = ["(_ bv%d %d)" % (i, self.tagsz) for i in range(self.xlen)]

        else:
            self.cursor = ["(_ bv%d %d)" % (0, self.tagsz)] * self.xlen

            for i, (p, q) in enumerate(self.perm):
                if self.tagtype == "onehot":
                    self.cursor[p] = "(_ bv%d %d)" % (1 << i, self.tagsz)
                elif self.tagtype == "normal":
                    self.cursor[p] = "(_ bv%d %d)" % (i+1, self.tagsz)
                else:
                    assert 0

        self.cursor = self.define("inbits", self.xlen*self.tagsz, self.concat(self.cursor))
        self.inbits = self.cursor
        self.stage = 0

    def mkbv(self, name, sz):
        print("(declare-fun %s () (_ BitVec %d))" % (name, sz), file=self.smtfile)
        return name

    def define(self, name, sz, expr):
        if sz is None:
            print("(define-fun %s () Bool %s)" % (name, expr), file=self.smtfile)
        else:
            print("(define-fun %s () (_ BitVec %d) %s)" % (name, sz, expr), file=self.smtfile)
        return name

    def constrain(self, expr):
        print("(assert %s)" % expr, file=self.smtfile)

    def extract(self, word, idx, sz):
        return "((_ extract %d %d) %s)" % (idx*sz+sz-1, idx*sz, word)

    def bitfield(self, word, idx, sz):
        return "((_ extract %d %d) %s)" % (idx+sz-1, idx, word)

    def getbit(self, word, idx):
        return "(= ((_ extract %d %d) %s) (_ bv1 1))" % (idx, idx, word)

    def concat(self, bits):
        return "(concat %s)" % " ".join(reversed(bits))

    def ite(self, cond, a, b):
        return "(ite %s %s %s)" % (cond, a, b)

    def eq(self, var, const, sz):
        return "(= %s (_ bv%d %d))" % (var, const, sz)

    def ne(self, var, const, sz):
        return "(distinct %s (_ bv%d %d))" % (var, const, sz)

    def lt(self, var, const, sz):
        return "(bvult %s (_ bv%d %d))" % (var, const, sz)

    def bfly(self, x, k):
        bits = [i for i in range(self.xlen)]

        a = 1 << k
        b = 2*a

        for j in range(self.xlen//2):
            p = b*(j//a) + j%a
            q = p + a;
            bits[p], bits[q] = bits[q], bits[p]

        return self.concat([self.extract(x, bits[i], self.tagsz) for i in range(self.xlen)])

    def gzip(self, x, k):
        bits = [self.extract(x, i, self.tagsz) for i in range(self.xlen)]
        new_bits = [None] * self.xlen
        for i in range(self.xlen):
            j = i & ~(3 << k)
            j |= (i & (1 << k)) << 1
            j |= (i & (2 << k)) >> 1
            new_bits[i] = bits[j]
        return self.concat(new_bits)

    def addcommand(self, rori=False, grevi=False, gzip=False, set_cmd=None, set_arg=None):
        cmd = self.mkbv("cmd%d" % self.stage, 2)
        arg = self.mkbv("arg%d" % self.stage, self.log2_xlen)

        if set_cmd is not None:
            self.constrain(self.eq(cmd, set_cmd, 2))

        if set_arg is not None:
            self.constrain(self.eq(arg, set_arg, self.log2_xlen))

        word = self.cursor
        k = 0

        if rori:
            op_word = self.cursor

            for i in range(self.log2_xlen):
                p = self.tagsz * (1 << i)
                new_word = "(concat ((_ extract %d 0) %s) ((_ extract %d %d) %s))" % (p-1, op_word, (self.xlen*self.tagsz)-1, p, op_word)
                op_word = self.define("w%d_%d" % (self.stage, k), self.xlen*self.tagsz, self.ite(self.getbit(arg, i), new_word, op_word))
                k += 1

            word = self.define("w%d_%d" % (self.stage, k), self.xlen*self.tagsz, self.ite(self.eq(cmd, 0, 2), op_word, word))
            k += 1

            self.constrain("(=> %s %s)" % (self.eq(cmd, 0, 2), self.ne(arg, 0, self.log2_xlen)))
        else:
            self.constrain(self.ne(cmd, 0, 2))

        if grevi:
            op_word = self.cursor

            for i in range(self.log2_xlen):
                op_word = self.define("w%d_%d" % (self.stage, k), self.xlen*self.tagsz, self.ite(self.getbit(arg, i), self.bfly(op_word, i), op_word))
                k += 1

            word = self.define("w%d_%d" % (self.stage, k), self.xlen*self.tagsz, self.ite(self.eq(cmd, 1, 2), op_word, word))
            k += 1

            self.constrain("(=> %s %s)" % (self.eq(cmd, 1, 2), self.ne(arg, 0, self.log2_xlen)))
        else:
            self.constrain(self.ne(cmd, 1, 2))

        if gzip:
            zipped_word = self.cursor
            unzipped_word = self.cursor

            for i in range(self.log2_xlen-1):
                zipped_word = self.define("w%d_%d" % (self.stage, k), self.xlen*self.tagsz, self.ite(self.getbit(arg, self.log2_xlen-i-1), self.gzip(zipped_word, self.log2_xlen-i-2), zipped_word))
                unzipped_word = self.define("w%d_%d" % (self.stage, k+1), self.xlen*self.tagsz, self.ite(self.getbit(arg, i+1), self.gzip(unzipped_word, i), unzipped_word))
                k += 2

            word = self.define("w%d_%d" % (self.stage, k), self.xlen*self.tagsz, self.ite("(and %s (not %s))" % (self.eq(cmd, 2, 2), self.getbit(arg, 0)), zipped_word, word))
            word = self.define("w%d_%d" % (self.stage, k+1), self.xlen*self.tagsz, self.ite("(and %s %s)" % (self.eq(cmd, 2, 2), self.getbit(arg, 0)), unzipped_word, word))
            k += 2

            for i in range(self.xlen):
                if not re.match(r"^0*(10+|11+0*[01])$", format(i, "0%db" % self.log2_xlen)):
                    self.constrain("(=> %s (not %s))" % (self.eq(cmd, 2, 2), self.eq(arg, i, self.log2_xlen)))
        else:
            self.constrain(self.ne(cmd, 2, 2))

        # some symmetry breaking
        if self.stage > 0:
            last_cmd = "cmd%d" % (self.stage-1)

            # all NOPs at the end
            self.constrain("(=> %s %s)" % (self.eq(last_cmd, 3, 2), self.eq(cmd, 3, 2)))

            # No double-rori
            self.constrain("(=> %s %s)" % (self.eq(last_cmd, 0, 2), self.ne(cmd, 0, 2)))

            # No double-grevi
            self.constrain("(=> %s %s)" % (self.eq(last_cmd, 1, 2), self.ne(cmd, 1, 2)))

            # No grevi-after-zip (canonically one grevi first, then zip)
            self.constrain("(=> %s %s)" % (self.eq(last_cmd, 2, 2), self.ne(cmd, 1, 2)))

        self.cursor = word
        self.stage += 1

    def solve(self):
        if self.order_mode:
            bits = []
            
            for i in range(self.xlen):
                tag = self.extract(self.cursor, i, self.tagsz)

                if i > 0:
                    tag = self.ite(self.eq(tag, 0, self.tagsz), bits[i-1], tag)

                tag = self.define("t%d" % i, self.tagsz, tag)
                bits.append(tag)

                if i > 0:
                    self.constrain("(not (bvult %s %s))" % (tag, bits[i-1]))

        else:
            for i, (p, q) in enumerate(self.perm):
                if self.tagtype == "onehot":
                    self.constrain(self.eq(self.extract(self.cursor, q, self.tagsz), 1 << i, self.tagsz))
                elif self.tagtype == "dense":
                    self.constrain(self.eq(self.extract(self.cursor, q, self.tagsz), p, self.tagsz))
                elif self.tagtype == "normal":
                    self.constrain(self.eq(self.extract(self.cursor, q, self.tagsz), i+1, self.tagsz))
                else:
                    assert 0

        for i in range(self.xlen):
            self.define("i%d" % i, self.tagsz, self.extract(self.inbits, i, self.tagsz))
            self.define("o%d" % i, self.tagsz, self.extract(self.cursor, i, self.tagsz))

        print("(check-sat)", file=self.smtfile)

        for i in range(self.xlen):
            print("(get-value (i%d))" % i, file=self.smtfile)
            print("(get-value (o%d))" % i, file=self.smtfile)

        for i in range(self.stage):
            print("(get-value (cmd%d))" % i, file=self.smtfile)
            print("(get-value (arg%d))" % i, file=self.smtfile)

        self.smtfile.close()

        if self.verbose:
            print("Running boolector.")

        start_time = time.time()
        if not self.dummy:
            os.system("boolector %s/progsynth.smt2 > %s/boolector.out" % (self.tempdir, self.tempdir))
        self.seconds = time.time() - start_time

        with open("%s/boolector.out" % self.tempdir, "r") as f:
            for line in f:
                line = line.split()
                if len(line) == 1 and line[0] == "sat":
                    self.sat = True

        if self.verbose:
            print("Solving time: %d:%02d (%s)" % (self.seconds // 60, self.seconds % 60, "sat" if self.sat else "unsat"))

        if not self.sat:
            return

        def decode_val(s, w):
            if False:
                print("val(%d) %s" % (w, s))
            if s.startswith("#b"):
                assert len(s) == w+2
                return int(s[2:], 2)
            assert 0

        with open("%s/boolector.out" % self.tempdir, "r") as f:
            self.mask = 0
            self.trace = list()
            if self.verbose:
                print("Permutation trace:")
            for line in f:
                line = line.split()
                if line[0].startswith("((i"):
                    ival = decode_val(line[1][0:-2], self.tagsz)
                if line[0].startswith("((o"):
                    oval = decode_val(line[1][0:-2], self.tagsz)
                    if self.verbose:
                        print("  %3d %3d %3d" % (len(self.trace), ival, oval))
                    if oval != 0:
                        self.mask |= 1 << len(self.trace)
                    self.trace.append((ival, oval))
            if self.verbose:
                print("mask: 0x%08x" % self.mask)

        def print_cmd(cmd, arg):
            if self.verbose:
                print("  ; cmd=%d arg=%d" % (cmd, arg))

            if cmd == 0:
                cmdname = "rori %d" % arg
                self.cprog.append("a0 = ror(a0, %d);" % arg)
                self.aprog.append("rori a0, a0, %d" % (arg))

            elif cmd == 1:
                cmdname = "grevi %d" % arg
                self.cprog.append("a0 = grev(a0, %d);" % arg)
                self.aprog.append("grevi a0, a0, %d" % (arg))

            elif cmd == 2:
                cmdname = "gzip %d" % arg
                self.cprog.append("a0 = gzip(a0, %d);" % arg)
                self.aprog.append("gzip a0, a0, %d" % (arg))

            else:
                cmdname = "nop"

            if self.verbose:
                print("  " + cmdname)

            if cmdname != "nop":
                self.prog.append(cmdname)

        self.prog = list()
        self.cprog = list()
        self.aprog = list()

        if self.verbose:
            print("Generated program:")

        with open("%s/boolector.out" % self.tempdir, "r") as f:
            for line in f:
                line = line.split()
                if line[0].startswith("((cmd"):
                    info_cmd = decode_val(line[1][0:-2], 2)
                if line[0].startswith("((arg"):
                    info_arg = decode_val(line[1][0:-2], self.log2_xlen)
                    print_cmd(info_cmd, info_arg)

if __name__ == "__main__":
    perm = [
        (0, 0),
        (1, 2),
        (2, 4),
        (3, 6),
    ]
    syn = Syn(32, perm, tempdir=".")
    syn.addcommand(rori=True, grevi=True, gzip=True)
    syn.solve()
