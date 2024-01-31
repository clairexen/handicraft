#!/usr/bin/env python3

# 8x8font.png is the "8x8 Bitmapped Font" by John Hall
# License: This font is free to use for any purpose.
# http://overcode.yak.net/12

from PIL import Image

im = Image.open("8x8font.png")
pix = im.load()

# shorten '-'
pix[ord('-')*8 + 1, 3] = 1
pix[ord('-')*8 + 2, 3] = 1

print("uint8_t fontmem [8*128] = {");
for i in range(128):
    for j in range(8):
        bits = 0
        for k in range(8):
            if pix[i*8+k, j]:
                bits = 2*bits
            else:
                bits = 2*bits + 1
        print("%3d," % bits, end="")
    if i >= 32 and i < 127:
        print(" // '%c'" % chr(i))
    else:
        print(" // %d" % i)
print("};")

print("uint8_t fontleft [128] = {");
for i in range(128):
    val = None
    for j in range(8):
        bits = 0
        for k in range(8):
            if not pix[i*8+k, j]:
                if val is None:
                    val = k
                else:
                    val = min(val, k)
    if val is None:
        val = 0
    print("%d," % val, end="" if (i+1) % 32 else "\n")
print("};")

print("uint8_t fontright [128] = {");
for i in range(128):
    val = None
    for j in range(8):
        bits = 0
        for k in range(8):
            if not pix[i*8+k, j]:
                if val is None:
                    val = k+1
                else:
                    val = max(val, k+1)
    if val is None:
        val = 4
    print("%d," % val, end="" if (i+1) % 32 else "\n")
print("};")

