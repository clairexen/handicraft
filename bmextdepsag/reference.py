#!/usr/bin/env python3
#
#  Reference implementation for Claire's SAG algorithm
#
#  Copyright (C) 2021  Claire Xenia Wolf <claire@clairexen.net>
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

import random
import sys

log2nbits = 5
TrueChars = "1ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FalseChars = "0abcdefghijklmnopqrstuvwxyz"


def rev(data):
    return "".join(reversed(data))


def mkctrl(mask):
    ctrl = ""
    carry, carry_n = "0", "1"
    for i in range(len(mask)//2):
        a, b = mask[2*i:2*(i+1)]
        a, b = a in TrueChars, b in TrueChars
        if a:
            ctrl += carry
        else:
            ctrl += carry_n
        if a != b:
            carry, carry_n = carry_n, carry
    return ctrl


def stage(data, ctrl):
        assert 2*len(ctrl) == len(data)
        a, b = "", ""
        for i in range(len(ctrl)):
            a += data[2*i + (1 if ctrl[i] in TrueChars else 0)]
            b += data[2*i + (0 if ctrl[i] in TrueChars else 1)]
        return a, b


def sag(data, mask, revmode=False):
    """Reference implementation of recursive SAG algorithm."""

    assert len(data) == len(mask)
    assert bin(len(data)).count("1") == 1

    inv = False
    rol = False
    ror = False

    if len([None for c in mask if c in TrueChars]) % 2 == 0:
        if not revmode:
            inv = True
            rol = True
            ror = mask[-1] in FalseChars
    else:
        if revmode:
            inv = True

    if inv:
        mask = mask[0:-1] + ("0" if mask[-1] in TrueChars else "1")

    if rol:
        data = data[-1] + data[0:-1]
        mask = mask[-1] + mask[0:-1]

    ctrl = mkctrl(mask)
    m1, m2 = stage(mask, ctrl)
    d1, d2 = stage(data, ctrl)

    if ror:
        m1, m2 = m2, m1[1:] + "0"
        d1, d2 = d2, d1[1:] + d1[0]

    if len(d1) > 1:
        d1 = sag(d1, m1, revmode=revmode)
        d2 = sag(d2, m2, revmode=revmode)

    result = "".join([a+b for a, b in zip(d1, d2)])

    return result


def runtests():
    """Simple test bench for the SAG reference implementations."""

    errcount=0

    for i in range((1 << log2nbits) + 1):
        chrs_up = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chrs_lo = "abcdefghijklmnopqrstuvwxyz"
        expect_up = ""
        expect_lo = ""
        data = ""
        mask = ""

        setbits = set(random.sample(range(1 << log2nbits), i))

        for k in range(1 << log2nbits):
            if k in setbits:
                mask = "1" + mask
                data = chrs_up[0] + data
                expect_up = chrs_up[0] + expect_up
                chrs_up = chrs_up[1:] + chrs_up[0]
            else:
                mask = "0" + mask
                data = chrs_lo[0] + data
                expect_lo = chrs_lo[0] + expect_lo
                chrs_lo = chrs_lo[1:] + chrs_lo[0]

        print(
            rev(data),
            rev(mask),
            mask.count("1")
        )

        if True:
            expect_sag = expect_up + expect_lo
            result_sag = sag(data, mask)

            if expect_sag != result_sag:
                errcount += 1

            print("  sag  ",
                rev(result_sag),
                rev(expect_sag),
                result_sag == expect_sag
            )

        if True:
            expect_sagr = expect_up + rev(expect_lo)
            result_sagr = sag(data, mask, revmode=True)

            if expect_sagr != result_sagr:
                errcount += 1

            print("  sagr ",
                rev(result_sagr),
                rev(expect_sagr),
                result_sagr == expect_sagr
            )

    print("Completed with %d errors." % errcount)
    return 1 if errcount != 0 else 0


if __name__ == "__main__":
    sys.exit(runtests())

