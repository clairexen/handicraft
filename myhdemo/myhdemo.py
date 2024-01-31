#!/usr/bin/env python3

from myhdl import *


class CounterIF:
    def __init__(self):
        self.clk = Signal(False)
        self.resetn = Signal(False)
        self.down = Signal(False)
        self.cnt = Signal(intbv()[8:0])


def IncDec(current_val, next_inc, next_dec):
    """ Increment/Decrement Generator """

    @always_comb
    def logic():
        next_inc.next = current_val + 1
        next_dec.next = current_val - 1

    return logic


def Counter(ports):
    """ A simple up/down counter. """

    next_inc = Signal(modbv()[8:0])
    next_dec = Signal(modbv()[8:0])

    incdec = IncDec(ports.cnt, next_inc, next_dec)

    @always(ports.clk.posedge)
    def logic():
        if not ports.resetn:
            ports.cnt.next = 0
        elif ports.down:
            ports.cnt.next = next_dec
        else:
            ports.cnt.next = next_inc

    return incdec, logic


def CounterTest():
    """ Testbench for Counter """

    ports = CounterIF()
    uut = Counter(ports)

    @always(delay(5))
    def clkdriver():
        ports.clk.next = not ports.clk

    @instance
    def control():
        for i in range(10):
            yield ports.clk.posedge
        ports.resetn.next = True
        for i in range(10):
            yield ports.clk.posedge
        ports.down.next = True
        for i in range(5):
            yield ports.clk.posedge
        ports.resetn.next = False

    @always(ports.clk.posedge)
    def monitor():
        print("resetn=%.1s, down=%.1s, cnt=%s" % (ports.resetn, ports.down, ports.cnt))

    return uut, clkdriver, control, monitor


# Run Simulation
sim = Simulation(traceSignals(CounterTest))
sim.run(300)

# Run Synthesis
toVerilog(Counter, CounterIF())

