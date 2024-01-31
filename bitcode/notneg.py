#!/usr/bin/env python3

import re

prognames = ["aggregate", "bithacks", "perm_b32", "swartmps", "bzip2", "oggenc"]

class Prog:
    def __init__(self, filename):
        self.insns = list()
        self.addr2insn = dict()
        self.labels = set()

        self.fusecnt = dict()
        self.upcnt = 0
        self.downcnt = 0
        self.bothcnt = 0
        self.nonecnt = 0
        self.matchcnt = 0

        with open(filename) as f:
            for line in f:
                match = re.match(r"^([0-9a-z]+) <\S+>:", line);
                if match:
                    self.labels.add(int(match.group(1), 16))
                    continue

                match = re.match(r"^\s+([0-9a-z]+):\s+([0-9a-z]+)\s+(\S+)\s+(\S+)", line);
                if match:
                    addr = int(match.group(1), 16)
                    opcode = int(match.group(2), 16)
                    op = match.group(3)
                    args = tuple(match.group(4).split(","))
                    if op in ("sw", "sh", "sb"):
                        args = ("zero",) + args
                    self.insns.append((addr, opcode, op, args))
                    self.addr2insn[addr] = len(self.insns)-1
                    continue

    def section(self, index):
        begin, end = index, index+1
        gotsrc = False

        inreg = self.insns[index][3][1]
        outreg = self.insns[index][3][0]

        while begin > 0:
            if self.insns[begin][0] in self.labels:
                break

            if begin == index:
                begin -= 1
                continue

            if self.insns[begin][3][0] == inreg:
                gotsrc = True
                break

            begin -= 1

        while end < len(self.insns):
            if self.insns[end][2] in ("ret", "jr", "j", "jal", "jalr", "call"):
                break

            if self.insns[end][3][0] == outreg:
                end += 1
                break

            end += 1

        return begin, end, gotsrc

    def count_sinks(self, begin, end):
        count = 0;
        sink = None

        outreg = self.insns[begin][3][0]

        for i in range(begin+1, end):
            if outreg in self.insns[i][3][1:]:
                sink = i
                count += 1
            if outreg == self.insns[i][3][0]:
                break

        return count, sink

    def analyze(self, index, verbose):
        if verbose:
            print()
            print("---------------------------------------------------")
            print()

        begin, end, gotsrc = self.section(index)

        if verbose:
            for k in range(begin, end):
                addr, opcode, op, args = self.insns[k]
                print("%s%6x %08x %6s %s" % ("* " if k == index else "  ", addr, opcode, op, args))
            print()

        source_sinks = 0
        if gotsrc:
            source_sinks, _ = self.count_sinks(begin, end)
            if verbose:
                print("Source found, %d sinks." % source_sinks)
        elif verbose:
            print("Source not found.")

        this_sinks, this_sink = self.count_sinks(index, end)

        if verbose:
            print("Found %d sinks for selected insn." % this_sinks)

        if source_sinks == 1:
            fused = "%s+%s" % (self.insns[begin][2], self.insns[index][2])
            if verbose:
                print("UPFUSE: %s" % fused)
            if fused not in self.fusecnt:
                self.fusecnt[fused] = 0
            self.fusecnt[fused] += 1
            self.upcnt += 1

        if this_sinks == 1:
            fused = "%s+%s" % (self.insns[index][2], self.insns[this_sink][2])
            if verbose:
                print("DOWNFUSE: %s" % fused)
            if fused not in self.fusecnt:
                self.fusecnt[fused] = 0
            self.fusecnt[fused] += 1
            self.downcnt += 1

        if source_sinks == 1 and this_sinks == 1:
            self.bothcnt += 1

        if source_sinks == 0 and this_sinks == 0:
            self.nonecnt += 1

        self.matchcnt += 1

    def analyze_op(self, match_op, verbose):
        for index in range(len(self.insns)):
            addr, opcode, op, args = self.insns[index]
            if op == match_op:
                self.analyze(index, verbose)


for match_op in ["not", "neg"]:
    print()
    print()
    print()
    print("Analysis of fusing options for '%s' instructions:" % match_op)
    print("==================================================")

    progs = dict()
    up_fused_ops = set()
    down_fused_ops = set()

    for name in prognames:
        progs[name] = Prog(name + ".dump")
        progs[name].analyze_op(match_op, False)

        for fused_op in progs[name].fusecnt.keys():
            if fused_op.startswith(match_op + "+"):
                down_fused_ops.add(fused_op)
            else:
                up_fused_ops.add(fused_op)

    def print_header():
        print("| %10s" % "", end="")
        for name in prognames:
            print(" | %10s" % name, end="")
        print(" |")

        print("|-%10s" % ("-" * 10), end="")
        for name in prognames:
            print("-+-%10s" % ("-" * 10), end="")
        print("-|")

    def print_data(t, f, nothird = False):
        print("| %10s" % t, end="")
        for name in prognames:
            print(" | %10d" % f(progs[name]), end="")
        print(" |")

        print("| %10s" % "", end="")
        for name in prognames:
            print(" | %8.3f %%" % (100 * f(progs[name]) / len(progs[name].insns)), end="")
        print(" |")

        if not nothird:
            print("| %10s" % "", end="")
            for name in prognames:
                print(" | %8.3f %%" % (0 if progs[name].matchcnt == 0 else 100 * f(progs[name]) / progs[name].matchcnt), end="")
            print(" |")

        print("|-%10s" % ("-" * 10), end="")
        for name in prognames:
            print("-+-%10s" % ("-" * 10), end="")
        print("-|")

    print()
    print("Percentage of %s instructions and percentage of those that can be fused" % match_op)
    print("with a previous instruction (up fusion) or a later instructions (down fusion):")

    print()
    print_header()
    print_data("insns", lambda p: len(p.insns), True)
    print_data(match_op, lambda p: p.matchcnt)
    print_data("fuse up", lambda p: p.upcnt)
    print_data("fuse down", lambda p: p.downcnt)
    print_data("fuse both", lambda p: p.bothcnt)
    print_data("fuse none", lambda p: p.nonecnt)

    print()
    print("Detailed analysis of up fusion options:")

    print()
    print_header()
    for fused_op in sorted(up_fused_ops):
        print_data(fused_op, lambda p: p.fusecnt[fused_op] if fused_op in p.fusecnt else 0)

    print()
    print("Detailed analysis of down fusion options:")

    print()
    print_header()
    for fused_op in sorted(down_fused_ops):
        print_data(fused_op, lambda p: p.fusecnt[fused_op] if fused_op in p.fusecnt else 0)

    print()
    print("Legend:")
    print("  1st line = number of instructions")
    print("  2nd line = percentage of total number of instructions in program")
    print("  3rd line = percentage of number of %s instructions in program" % match_op)

