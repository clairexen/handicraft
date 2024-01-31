#!/usr/bin/env python3

from Verilog_VCD.Verilog_VCD import parse_vcd

ops_bits = None
args_bits = None

for netinfo in parse_vcd("permsolve/engine_0/trace0.vcd").values():
    for net in netinfo['nets']:
        if net["hier"] == "permcluster" and net["name"] == "ops":
            ops_bits = netinfo['tv'][0][1]
        if net["hier"] == "permcluster" and net["name"] == "args":
            args_bits = netinfo['tv'][0][1]

assert ops_bits is not None
assert args_bits is not None

ops = reversed(list(zip(
    [int(ops_bits[n:n+2], 2) for n in range(0, len(ops_bits), 2)],
    [int(args_bits[n:n+5], 2) for n in range(0, len(args_bits), 5)]
)))

for op, arg in ops:
    if op == 0:
        print("ROR(%d)" % arg)
    elif op == 1:
        print("GREV(%d)" % arg)
    elif op == 2:
        print("SHFL(%d)" % (arg & 15))
    elif op == 3:
        print("UNSHFL(%d)" % (arg & 15))
    else:
        assert 0
