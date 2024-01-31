#!/usr/bin/env python3
for i in range(512):
    v = ((i << 3) + (i >> 3)) & 255
    print("ff" if v == 0 else "%02x" % (v & 255))
