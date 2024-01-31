#!/usr/bin/env python3
#
#  Copyright (C) 2017  Clifford Wolf <clifford@clifford.at>
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

from sys import argv, stdout
from subprocess import Popen, PIPE
import numpy as np

with open("data.txt", "w") as f:
    for N in range(1, 33):
        print("Testing patterns with %2d bits: " % N, end="")
        stdout.flush()

        for k in range(50):
            from_bits = np.arange(32)
            to_bits = np.arange(32)

            np.random.shuffle(from_bits)
            np.random.shuffle(to_bits)

            from_bits = from_bits[0:N]
            to_bits = to_bits[0:N]

            pattern = ["-" for i in range(32)]
            symbols = "0123456789abcdefghijklmnopqrstuv"

            for i in range(N):
                pattern[31-from_bits[i]] = symbols[to_bits[i]]

            pattern = "".join(pattern)
            print("%d" % N, end="", file=f)
            for mode in ["-g -b", "-b", "-s", ""]:
                data = -1
                with Popen(("./permsyn -d %s %s" % (mode, pattern)).split(), stdout=PIPE) as proc:
                    for line in proc.stdout.readlines():
                        data = int(line.decode("ascii"))
                print(" %d" % data, end="", file=f)
            print("", file=f)
            f.flush()

            print(".", end="")
            stdout.flush()

        print()

