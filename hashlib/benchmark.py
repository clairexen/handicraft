#!/usr/bin/python

from __future__ import division
from __future__ import print_function

from matplotlib import pyplot as plt
import numpy as np
import sys

plot_megabytes=False

data = list()
comp_s = set()
impl_s = ["map", "set", "unordered_map", "unordered_set", "dict", "pool"]
type_s = set()
mode_s = set()
size_s = set()

nice_label = {
    "map" : "std::map",
    "set" : "std::set",
    "unordered_map" : "std::unordered_map",
    "unordered_set" : "std::unordered_set",
    "dict" : "hashlib::dict",
    "pool" : "hashlib::pool"
}

nice_color = {
    "map" : "red",
    "set" : "darkred",
    "unordered_map" : "blue",
    "unordered_set" : "darkblue",
    "dict" : "green",
    "pool" : "darkgreen"
}

with open('benchmark.dat', 'r') as f:
    for line in f:
        dat = line.split();
        data.append(dat)
        comp_s.add(dat[0])
        # impl_s.add(dat[1])
        type_s.add(dat[2])
        mode_s.add(dat[3])
        size_s.add(dat[4])

plt.xkcd()
if plot_megabytes:
    fig = plt.figure(figsize=(10, 6*4))
else:
    fig = plt.figure(figsize=(10, 3*4))

def add_benchmark_plot(c, t, m):
    if t == "int" and m == "dense": row = 0
    if t == "int" and m == "sparse": row = 1
    if t == "string" and m == "dense": row = 2
    if t == "string" and m == "sparse": row = 3

    for sp in ([1, 2] if plot_megabytes else [1]):
        ax = fig.add_subplot(8 if plot_megabytes else 4, 2, (1 if c == "clang" else 2) + (4*row + 2*sp - 2 if plot_megabytes else 2*row))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plots = list()

        for i in impl_s:
            if i != "none":
                p_size = list()
                p_time = dict()
                p_memory = dict()

                for dat in data:
                    if dat[0] == c and dat[1] == i and dat[2] == t and dat[3] == m:
                        p_time[int(dat[4])] = float(dat[7])
                        p_memory[int(dat[4])] = float(dat[8])
                        p_size.append(int(dat[4]))

                for dat in data:
                    if dat[0] == c and dat[1] == "none" and dat[2] == t and dat[3] == m:
                        p_time[int(dat[4])] -= float(dat[7])
                        p_memory[int(dat[4])] -= float(dat[8])

                if sp == 1:
                    plt.semilogx(p_size, [ p_time[s] for s in p_size ], label=nice_label[i], color=nice_color[i])
                else:
                    plt.loglog(p_size, [ p_memory[s] for s in p_size ], label=nice_label[i], color=nice_color[i])

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(10) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(10) 

        if sp == 1:
            plt.legend(loc=2, fontsize=7, borderaxespad=2)
            if c == "clang":
                plt.ylabel('seconds')
            if not plot_megabytes and row == 3:
                plt.xlabel('container size')
            plt.title('%s - %s - %s' % (c, t, m))
        else:
            if c == "clang":
                plt.ylabel('megabytes')
            if row == 3:
                plt.xlabel('container size')


for c in comp_s:
    for t in type_s:
        for m in mode_s:
            add_benchmark_plot(c, t, m)

plt.tight_layout()
plt.savefig('benchmark.png')

