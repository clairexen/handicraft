#!/usr/bin/env python3
#
# Experimentally measure the false positive rate of
# (relatively small) bloom filters
#
# Parameters:
#   X  ...  number of samples for measuring c and q
#   Y  ...  number of samples for measuring C
# Inputs:
#   k  ...  number of hash functions
#   m  ...  number of bits (or buckets)
#   n  ...  number of inserted elements
# Outputs:
#   q  ...  rate of mask bits still set to 0 and the end
#   e  ...  mean collision rate for the n elements so far
#   E  ...  mean collision rate on the next instert
#
# E is also the expected false positive rate when testing
# membership of out-of-set elements.

from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np

if True:
    Ks = (2, )
    Ms = (64, 128, 256)
    Ns = tuple(range(0, 50))
    
    enable_plot_E_for_multiple_k = True
    enable_plot_e_for_multiple_k = True
    enable_plot_q_for_multiple_k = True
    enable_plot_E_for_multiple_m = True
    enable_plot_e_for_multiple_m = True
    enable_plot_q_for_multiple_m = True
    
    X = 200
    Y = 200

else:
    Ks = (2, )
    Ms = (64, 128, 256, 512)
    Ns = tuple(range(0, 150, 5))
    
    enable_plot_E_for_multiple_k = False
    enable_plot_e_for_multiple_k = False
    enable_plot_q_for_multiple_k = False
    enable_plot_E_for_multiple_m = True
    enable_plot_e_for_multiple_m = True
    enable_plot_q_for_multiple_m = True
    
    X = 50
    Y = 50
    
#%%

def test_filter(m, n, k):
    bitmask = 0

    e = 0
    for i in range(n):
        isCollision = True
        bits = [int(v) for v in np.random.randint(m, size=k)]
        for idx in bits:
            if ((bitmask >> idx) & 1) == 0:
                isCollision = False
            bitmask |= 1 << idx
        if isCollision:
            e += 1/n
        #print(f"At {i=}: {bitmask=:0{m}b}, {bits=}")
    q = f"{bitmask:0{m}b}".count("0") / m
    #print(f"{bitmask=:0{m}b} {m=} {q=} {type(bitmask)}")
    assert bitmask >= 0
    
    E = 0
    for i in range(Y):
        bits = [int(v) for v in np.random.randint(m, size=k)]
        for idx in bits:
            if ((bitmask >> idx) & 1) == 0:
                break
        else:
            E += 1 / Y
    
    return q, e, E

def test_filter_x(m, n, k, x = X):
    eValues = [test_filter(m,n,k) for i in range(x)]
    # print(eValues)I
    return np.mean(eValues, axis=0) #, np.std(eValues)

#%%

traces = SimpleNamespace()

for k in Ks:
    for m in Ms:
        trace_name = f"k{k}m{m}"
        print(f"[{trace_name}] Testing {X} iterations of all Ns..", end="", flush=True)
        trace = SimpleNamespace(q=[], e=[], E=[], n=Ns)
        setattr(traces, trace_name, trace)
        for n in Ns:
            print(".", end="", flush=True)
            qq, ee, EE = test_filter_x(m, n, k)
            trace.q.append(qq)
            trace.e.append(ee)
            trace.E.append(EE)
        print("", flush=True)

#%%

def plot_Eeq_for_multiple_k(item, kk=Ks, m=64):
    plt.title(f"m = number of bits in bitmask = {m} bits")
    for k in kk:
        plt.plot(Ns, getattr(getattr(traces, f"k{k}m{m}"), item), label=f"E_k={k}")
    plt.legend(loc="upper left")
    plt.xlabel("n = number of inserted items")
    if item == "E":
        plt.ylabel("E = probability of collision on next insert\n" + \
                   "= probability of false positive after n inserts")
    elif item == "e":
        plt.ylabel("e = avg ratio of colliions during first n insert\n")
    elif item == "q":
        plt.ylabel("q = ratio of bitmask bits that are stil clear\n" + \
                   "(zero) after n elements have been inserts")
    plt.ylim(0.0, 1.0)
    plt.show()

def plot_Eeq_for_multiple_m(item, k=2, mm=Ms):
    plt.title(f"k = number of hash functions = {k}")
    for m in mm:
        plt.plot(Ns, getattr(getattr(traces, f"k{k}m{m}"), item), label=f"E_m={m}")
    plt.legend(loc="upper left")
    plt.xlabel("n = number of inserted items")
    if item == "E":
        plt.ylabel("E = probability of collision on next insert\n" + \
                   "= probability of false positive after n inserts")
    elif item == "e":
        plt.ylabel("e = avg ratio of colliions during first n insert\n")
    elif item == "q":
        plt.ylabel("q = ratio of bitmask bits that are stil clear\n" + \
                   "(zero) after n elements have been inserts")
    plt.ylim(0.0, 1.0)
    plt.show()

#%%

if enable_plot_E_for_multiple_k:
    for m in Ms:
        plot_Eeq_for_multiple_k("E", kk=Ks, m=m)

if enable_plot_e_for_multiple_k:
    for m in Ms:
        plot_Eeq_for_multiple_k("e", kk=Ks, m=m)

if enable_plot_q_for_multiple_k:
    for m in Ms:
        plot_Eeq_for_multiple_k("q", kk=Ks, m=m)

#%%

if enable_plot_E_for_multiple_m:
    for k in Ks:
        plot_Eeq_for_multiple_m("E", k=k, mm=Ms)

if enable_plot_e_for_multiple_m:
    for k in Ks:
        plot_Eeq_for_multiple_m("e", k=k, mm=Ms)

if enable_plot_q_for_multiple_m:
    for k in Ks:
        plot_Eeq_for_multiple_m("q", k=k, mm=Ms)