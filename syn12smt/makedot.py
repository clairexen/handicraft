#!/usr/bin/env python3

import re
import fileinput

database = dict()
maxlevel = 0

for line in fileinput.input():
    res = re.match(r"^ \(\(f(\d+)([ab])([^)]*)\) #b([^)]+)\)", line)
    if res:
        level, ab, path, value = res.groups()
        label = "%s%s%s" % (level, path.replace(" ", ""), ab)
        maxlevel = max(maxlevel, int(level))
        database[label] = value

def mask2list(bits):
    v = ""
    for i in range(len(bits)):
        if bits[i] == '1':
            v = "%d%s%s" % (len(bits)-i, "" if v == "" else " ", v)
    return v

def generate_nodes(level, path):
    if level == maxlevel:
        label_a = "%d%sa" % (level, path)
        value_a = database[label_a]
        print("n%s [label=\"%s\"];" % (path, mask2list(value_a)))
    else:
        label_a = "%d%sa" % (level, path)
        label_b = "%d%sb" % (level, path)
        value_a = database[label_a]
        value_b = database[label_b]
        print("n%s [shape=box,label=\"%s ⚖️ %s\"];" % (path, mask2list(value_a), mask2list(value_b)))
        generate_nodes(level+1, path+"FF")
        generate_nodes(level+1, path+"FT")
        generate_nodes(level+1, path+"TF")

def generate_edges(level, path):
    if level < maxlevel:
        print("n%s -> n%sFT [label=\">\"];" % (path, path))
        print("n%s -> n%sFF [label=\"=\"];" % (path, path))
        print("n%s -> n%sTF [label=\"<\"];" % (path, path))
        generate_edges(level+1, path+"FF")
        generate_edges(level+1, path+"FT")
        generate_edges(level+1, path+"TF")

print("digraph G {")
print("rankdir=LR;")
print("ordering=out;")
generate_nodes(0, "")
generate_edges(0, "")
print("}")
