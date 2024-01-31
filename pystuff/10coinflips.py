#!/usr/bin/env python3
# https://twitter.com/oe1cxw/status/714944851125649408

import numpy as np
from numba import jit, u4, u8, b1
from collections import defaultdict

@jit(nopython=True)
def single_run_v2():
    last_result = np.random.randint(2)
    for i in range(1, 10):
        this_result = np.random.randint(2)
        if this_result == last_result:
            return (False, u4(i+1))
        last_result = this_result
    return (True, u4(10))

@jit(nopython=True)
def single_run_v4():
    last_result2 = np.random.randint(2)
    last_result1 = np.random.randint(2)
    for i in range(2, 10):
        this_result = np.random.randint(2)
        if this_result != last_result2:
            return (False, u4(i+1))
        last_result2 = last_result1
        last_result1 = this_result
    return (True, u4(10))

@jit(nopython=True)
def complete_run():
    i = u4(0)
    j = u4(0)
    while True:
        r, c = single_run_v2()
        i += 1
        j += c
        if r: break
    return (i, j)

sum_i = 0
sum_j = 0

hist_i = defaultdict(int)
hist_j = defaultdict(int)

n = 500000
block = 10000
bucket_i = 25
bucket_j = 50
assert n % block == 0

for k in range(1, n+1):
    i, j = complete_run()
    hist_i[i // bucket_i] += 1
    hist_j[j // bucket_j] += 1
    sum_i += i
    sum_j += j
    if k % block == 0:
        print("%6d | %8.3f %8.3f | %5d %5d" % (k, sum_i / k, sum_j / k, i, j))

print()
print("-- hist_i --")
acc_i = 0
for i in hist_i:
    acc_i += hist_i[i]
    print("%5d .. %5d: %5d %6d %6.2f%% |%-50s| %4.2f%% %s" % (i * bucket_i, (i+1) * bucket_i - 1, hist_i[i], acc_i,
            100 * acc_i / n, "*" * int(50 * acc_i / n), 100 * hist_i[i] / n, "=" * int(500 * hist_i[i] / n)))
    if 100 * acc_i / n > 90: break

print()
print("-- hist_j --")
acc_j = 0
for j in hist_j:
    acc_j += hist_j[j]
    print("%5d .. %5d: %5d %6d %6.2f%% |%-50s| %4.2f%% %s" % (j * bucket_j, (j+1) * bucket_j - 1, hist_j[j], acc_j,
            100 * acc_j / n, "*" * int(50 * acc_j / n), 100 * hist_j[j] / n, "=" * int(500 * hist_j[j] / n)))
    if 100 * acc_j / n > 90: break

