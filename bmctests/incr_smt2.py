#!/usr/bin/python3

import sys, getopt
import subprocess

kmax = 20
top_module = "main"
debug_mode = True
model_mode = False

def usage():
    print("""
Usage: %s [options] smt2-file..

    -d
        print yices input/output (debug mode)

    -k N
        stop model checking after N time steps

    -t top_module
        set name of the top module (default="main")

    -m
        get model from yices
""" % (sys.argv[0]))
    sys.exit(1)

try:
    opts, args = getopt.getopt(sys.argv[1:], "dmk:t:")
except:
    usage()

for o, a in opts:
    if o == "-d":
        debug_mode = True
    elif o == "-m":
        model_mode = True
    elif o == "-k":
        kmax = int(a)
    elif o == "-t":
        top_module = a
    else:
        usage()

if len(args) == 0:
    usage()

yices = subprocess.Popen(['yices-smt2', '--incremental'], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def yices_write(stmt):
    stmt = stmt.strip()
    if debug_mode:
        print("> %s" % stmt)
    yices.stdin.write(bytes(stmt + "\n", "ascii"))
    yices.stdin.flush()

def yices_read():
    stmt = []
    count_brackets = 0

    while True:
        line = yices.stdout.readline().decode("ascii").strip()
        count_brackets += line.count("(")
        count_brackets -= line.count(")")
        stmt.append(line)
        if debug_mode:
            print("< %s" % line)
        if count_brackets == 0: break

    stmt = "".join(stmt)
    if stmt.startswith("(error"):
        print("Yices Error: %s" % stmt, file=sys.stderr)
        sys.exit(1)

    return stmt

yices_write("(set-logic QF_AUFBV)")

for fn in args:
    with open(fn, "r") as f:
        for line in f:
            yices_write(line)

for k in range(kmax):
    print("Checking sequence of length %d.." % k)

    yices_write("(declare-fun s%d () %s_s)" % (k, top_module))
    if k == 0:
        yices_write("(assert (%s_i s%d))" % (top_module, k))
    else:
        yices_write("(assert (%s_t s%d s%d))" % (top_module, k-1, k))

    yices_write("(push 1)")
    yices_write("(assert (not (%s_a s%d)))" % (top_module, k))

    yices_write("(check-sat)")
    result = yices_read()

    if result == "sat":
        print("Found model -> proof failed!")
        if model_mode:
            yices_write("(get-model)")
            print(yices_read())
        sys.exit(0)

    yices_write("(pop 1)")
    yices_write("(assert (%s_a s%d))" % (top_module, k))

print("Finished BMC. No models found. SUCCESS!")

