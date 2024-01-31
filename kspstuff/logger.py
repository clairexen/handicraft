#!/usr/bin/env python3

import krpc
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

#%% Collect data

if True:
    conn = krpc.connect()
    vessel = conn.space_center.active_vessel
    
    data = list()
    
    while len(data) < 1000:
        data.append((
            vessel.orbit.apoapsis,
            vessel.orbit.periapsis,
            vessel.orbit.eccentricity,
            vessel.orbit.time_to_apoapsis,
            vessel.orbit.time_to_periapsis
        ))
        print(len(data))
        sleep(0.1)
    
    data = np.array(data)
    np.savetxt("logger.dat", data)

else:
    data = np.loadtxt("logger.dat")

#%% Plot data

plt.figure()
plt.plot(data[:, 0])

plt.figure()
plt.plot(data[:, 0])
plt.plot(data[:, 1])

plt.figure()
plt.plot(data[:, 2])

