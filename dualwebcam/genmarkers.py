#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import os
import sys
import Image

print('Generating 1024 marker bitmaps', end='', file=sys.stderr)

print('var marker_bitmaps = [')
for i in range(1024):
    os.system('aruco-1.2.4/utils/aruco_create_marker {} tmpfile.ppm 7'.format(i))
    im = Image.open('tmpfile.ppm')
    print('[', end='')
    for y in range(1,6):
        bitmap = 0
        for x in range(1,6):
            pix = im.getpixel((x,y))
            if pix > 255/2:
                bitmap = bitmap | (1 << (x-1))
        if y > 1:
            print(',', end='')
        print('{}'.format(str(bitmap).rjust(3)), end='')
    if i < 1023:
        thisend = ' '
        if i % 5 == 4:
            thisend = '\n'
        print('],', end=thisend)
    else:
        print(']')
    print('.', end='', file=sys.stderr)
print('];')

print(' done.', file=sys.stderr)

