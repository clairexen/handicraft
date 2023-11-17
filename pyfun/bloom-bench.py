#!/usr/bin/env python3
#
# Experimentally measure the false positive rate of
# (relatively small) bloom filters
#
# Parameters:
#   X  ...  number of samples for measuring q and e
# Inputs:
#   k  ...  number of hash functions
#   m  ...  number of bits (or buckets)
#   n  ...  number of inserted elements
# Outputs:
#   q  ...  rate of mask bits still set to 0 and the end
#   e  ...  mean collision rate for the n elements so far
#   E  ...  mean collision rate on next insert (1/q)^k
#
# E is also the expected false positive rate when testing
# membership of out-of-set elements.

from types import SimpleNamespace
import matplotlib.pyplot as plt
import random

if True:
    X = 10
    plot_ES = True
    Ks = (2, 3, 4)
    Ms = (64, 128, 256)
    Ns = tuple(range(0, 50))

    enable_plot_E_for_multiple_k = True
    enable_plot_e_for_multiple_k = True
    enable_plot_q_for_multiple_k = True

    enable_plot_E_for_multiple_m = True
    enable_plot_e_for_multiple_m = True
    enable_plot_q_for_multiple_m = True

else:
    X = 50
    plot_ES = True
    Ks = (2, )
    Ms = (64, 128, 256, 512)
    Ns = tuple(range(0, 150, 5))

    enable_plot_E_for_multiple_k = False
    enable_plot_e_for_multiple_k = False
    enable_plot_q_for_multiple_k = False

    enable_plot_E_for_multiple_m = True
    enable_plot_e_for_multiple_m = True
    enable_plot_q_for_multiple_m = True


#%%

def test_filter(m, n, k):
    all_bits = [ 1 << i for i in range(m) ]
    bitmask = 0

    e = 0
    for i in range(n):
        bits = sum(random.sample(all_bits, k))
        if (bitmask | bits) == bitmask:
            e += 1/n
        bitmask |= bits
        #print(f"At {i=}: {bitmask=:0{m}b}, {bits=}")

    q = f"{bitmask:0{m}b}".count("0") / m
    E = (1-q)**k

    #print(f"{bitmask=:0{m}b} {m=} {q=} {type(bitmask)}")
    assert bitmask >= 0

    if plot_ES:
        ES = 0
        for i in range(500):
            bits = sum(random.sample(all_bits, k))
            if (bitmask | bits) == bitmask:
                ES += 1 / 500
    else:
        ES = E

    return q, e, E, ES

def test_filter_x(m, n, k, x = X):
    vals = [test_filter(m,n,k) for i in range(x)]
    return [sum(v) / len(v) for v in zip(*vals)]

#%%

traces = SimpleNamespace()

for k in Ks:
    for m in Ms:
        trace_name = f"k{k}m{m}"
        print(f"[{trace_name}] Testing {X} iterations of all Ns..", end="", flush=True)
        trace = SimpleNamespace(q=[], e=[], E=[], ES=[], n=Ns)
        setattr(traces, trace_name, trace)
        for n in Ns:
            print(".", end="", flush=True)
            qq, ee, EE, ES = test_filter_x(m, n, k)
            trace.q.append(qq)
            trace.e.append(ee)
            trace.E.append(EE)
            trace.ES.append(ES)
        print("", flush=True)

#%%

def plot_Eeq_for_multiple_k(item, kk=Ks, m=64):
    plt.title(f"m = number of bits in bitmask = {m} bits")
    for k in kk:
        plt.plot(Ns, getattr(getattr(traces, f"k{k}m{m}"), item), label=f"{item}_k={k}")
    plt.legend(loc="upper left")
    plt.xlabel("n = number of inserted items")
    if item == "E":
        if plot_ES:
            for k in kk:
                plt.plot(Ns, getattr(traces, f"k{k}m{m}").ES, label=f"ES_k={k}")
        plt.ylabel("E = probability of collision on next insert\n" + \
                   "= probability of false positive after n inserts")
    elif item == "e":
        plt.ylabel("e = avg ratio of colliions during first n insert")
    elif item == "q":
        plt.ylabel("q = ratio of bitmask bits that are stil clear\n" + \
                   "(zero) after n elements have been inserts")
    plt.ylim(0.0, 1.0)
    plt.show()

def plot_Eeq_for_multiple_m(item, k=2, mm=Ms):
    plt.title(f"k = number of hash functions = {k}")
    for m in mm:
        plt.plot(Ns, getattr(getattr(traces, f"k{k}m{m}"), item), label=f"{item}_m={m}")
    plt.legend(loc="upper left")
    plt.xlabel("n = number of inserted items")
    if item == "E":
        if plot_ES:
            for m in mm:
                plt.plot(Ns, getattr(traces, f"k{k}m{m}").ES, label=f"ES_m={m}")
        plt.ylabel("E = probability of collision on next insert\n" + \
                   "= probability of false positive after n inserts")
    elif item == "e":
        plt.ylabel("e = avg ratio of colliions during first n insert")
    elif item == "q":
        plt.ylabel("q = ratio of bitmask bits that are stil clear\n" + \
                   "(zero) after n elements have been inserts")
    plt.ylim(0.0, 1.0)
    plt.show()


if enable_plot_E_for_multiple_k:
    for m in Ms:
        plot_Eeq_for_multiple_k("E", kk=Ks, m=m)

if enable_plot_e_for_multiple_k:
    for m in Ms:
        plot_Eeq_for_multiple_k("e", kk=Ks, m=m)

if enable_plot_q_for_multiple_k:
    for m in Ms:
        plot_Eeq_for_multiple_k("q", kk=Ks, m=m)


if enable_plot_E_for_multiple_m:
    for k in Ks:
        plot_Eeq_for_multiple_m("E", k=k, mm=Ms)

if enable_plot_e_for_multiple_m:
    for k in Ks:
        plot_Eeq_for_multiple_m("e", k=k, mm=Ms)

if enable_plot_q_for_multiple_m:
    for k in Ks:
        plot_Eeq_for_multiple_m("q", k=k, mm=Ms)
