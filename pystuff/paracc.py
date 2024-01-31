#!/usr/bin/env python3

import numpy as np

data_len_log2 = 5
data_len = 1 << data_len_log2

indata = np.random.randint(10, size=data_len)
# indata = np.arange(data_len)

outdata_simple = indata.copy()

for i in range(1, data_len):
    outdata_simple[i] += outdata_simple[i-1]

outdata_smart = indata.copy()
sumdata = indata.copy()

for p in range(data_len_log2):
    print()
    print("-- %d --" % p)
    
    for i in range(2**p, data_len, 2**(p+1)):
        s = sumdata[i-1]
        print("outdata[%2d:%2d] += (sum(indata[%2d:%2d]) = sumdata[%2d] = %2d)" % \
                (i, i+2**p, i-2**p, i, i-1, s))
        for k in range(2**p):
            outdata_smart[i+k] += s

    for i in range(2**(p+1) - 1, data_len, 2**(p+1)):
        print("sumdata[%2d] += (sumdata[%2d] = %2d), sumdata[%2d] = outdata[%2d]" % \
                (i, i - 2**p, sumdata[i - 2**p], i, i))
        sumdata[i] += sumdata[i - 2**p]
        assert sumdata[i] == outdata_smart[i]
        

print()
for i in range(data_len):
    print("%2d | %3d | %3d | %3d %3d %s" % (i, indata[i], sumdata[i], outdata_simple[i],
        outdata_smart[i],  "ok" if outdata_simple[i] == outdata_smart[i] else ""))

assert (outdata_simple == outdata_smart).all()
