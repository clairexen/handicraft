#!/usr/bin/env python3
#
# Usage Example:
#   python3 xorgrid.py > xorgrid.v
#   yosys -p 'synth_ice40 -top xorgrid' xorgrid.v

import click
import numpy as np
from numpy.random import randint

@click.command()
@click.option('-I', 'I', default=16, help="number of primary input bits")
@click.option('-O', 'O', default=16, help="number of primary output bits")
@click.option('-C', 'C', default=16, help="number of global control bits")   # increase for more complex routing
@click.option('-W', 'W', default=16, help="width of tile grid")              # increase for larger designs
@click.option('-H', 'H', default=16, help="height of tile grid")             # increase for larger designs
@click.option('-F', 'F', default=8, help="number of tile input bits")        # increase for larger designs with more complex routing
@click.option('-N', 'N', default=6, help="number of tile output bits")       # increase for larger designs with simpler routing
@click.option('-D', 'D', default=4, help="max distance")                     # increase for more complex routing
def xorgrid(I, O, C, W, H, F, N, D):
    print(f"""
module xorgrid (
  input clk,
  input rst,
  input [{I-1}:0] in,
  input [{C-1}:0] ctrl,
  output [{O-1}:0] out
);
""")

    lastreg = f"reg{W-1}x{H-1}[{N-1}]"
    for x in range(W):
        for y in range(H):
            print(f"  reg [{N-1}:0] reg{x}x{y};")
            print(f"  wire [{F-1}:0] fin{x}x{y};")
            print()
            print(f"  always @(posedge clk) begin")
            for idx in range(N):
                print(f"    reg{x}x{y}[{idx}] <= rst ? 1'b {randint(2)} : "
                      f"^{{{lastreg}, fin{x}x{y} & (ctrl[{randint(C)}] ? "
                      f"{F}'d {randint(1<<F)} : {F}'d {randint(1<<F)})}};")
                lastreg = f"reg{x}x{y}[{idx}]"
            print(f"  end")
            print()

    inbits = list(range(I)) + [None]*(W*H*F-I)
    np.random.shuffle(inbits)
    for x in range(W):
        for y in range(H):
            for idx in range(F):
                if (inbit := inbits.pop()) is not None:
                    print(f"  assign fin{x}x{y}[{idx}] = in[{inbit}];")
                else:
                    x2 = randint(max(x-D, 0), min(x+D, W))
                    y2 = randint(max(y-D, 0), min(y+D, H))
                    print(f"  assign fin{x}x{y}[{idx}] = reg{x2}x{y2}[{randint(N)}];")
            print()
    assert len(inbits) == 0

    for idx in range(O):
        print(f"  assign out[{idx}] = reg{randint(W)}x{randint(H)}[{randint(N)}];")
    print()

    print("endmodule")

if __name__ == '__main__':
    xorgrid()
