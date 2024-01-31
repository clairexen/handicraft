#!/usr/bin/env python3

import numpy as np
import numpy.random

db = []

for k in range(10000):
    samples = np.zeros(30)
    pulses = np.zeros(30)
    s = np.random.uniform(2, 3)
    for i in range(3):
        a = np.random.uniform(5, 10)
        p = 15.0 + np.random.normal()
        while pulses[int(p)-1] != 0 or pulses[int(p)] != 0 or pulses[int(p)+1] != 0 or pulses[int(p)+2] != 0:
            p = np.random.uniform(5, 25)
        for i in range(30):
            samples[i] += a * np.exp(-((i-p)**2) / (s**2))
        pulses[int(p)] += (1-p+int(p))
        pulses[int(p)+1] += (p-int(p))        
    db.append(np.hstack([samples, pulses]))

db = np.array(db)
np.savetxt('testdata.txt', db)
