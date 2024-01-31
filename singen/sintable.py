#!/usr/bin/env python3

import math

with open("sintable.hex", "wt") as f:
    for i in range(1024):
        print("%04x" % round(((1 << 15) - 0.5) * (1 + math.sin(2 * math.pi * i / 1024))), file=f)

