#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import argparse
import serial
import time
import sys

# see https://code.google.com/p/mojo-loader/source/browse/src/Loader.c
# see https://code.google.com/p/mojo-ide/source/browse/Mojo%20Loader/src/com/embeddedmicro/mojo/MojoLoader.java

def connect_and_reset(tty):
    dev = serial.Serial(tty, 115200, timeout=3)
    dev.setDTR(True)
    time.sleep(0.005)
    dev.setDTR(False)
    time.sleep(0.005)
    dev.setDTR(True)
    time.sleep(0.005)
    dev.flushInput()
    return dev

def clear_flash(tty):
    dev = connect_and_reset(tty)
    dev.write('E')

    response = dev.read(1)
    assert response == 'D'

    dev.close()

def send_binary(tty, filename, flash, verify):
    dev = connect_and_reset(tty)
    f = open(filename, 'rb')

    f.seek(0, 2)
    file_size = f.tell()

    if flash:
        if verify:
            dev.write('V')
        else:
            dev.write('F')
    else:
        dev.write('R')

    response = dev.read(1)
    assert response == 'R'

    dev.write(bytearray([ (file_size >> (i*8)) % 256 for i in range(4) ]))

    response = dev.read(1)
    assert response == 'O'

    f.seek(0)
    total = 0
    print('Uploading:   0%', end='', file=sys.stderr)
    while total < file_size:
        data = f.read(1024)
        rc = dev.write(data)
        assert rc == len(data)
        total += len(data)
        print('\rUploading: %3d%%' % (100*total // file_size), end='', file=sys.stderr)
    print('\rUploading: 100%', file=sys.stderr)

    response = dev.read(1)
    assert response == 'D'

    if flash and verify:
        dev.write('S')
        response = dev.read(1)
        assert response == '\xAA'

        flash_size = 0
        for i in range(4):
            response = bytearray(dev.read(1))[0]
            flash_size |= response << (i*8)
        assert flash_size == file_size + 5

        f.seek(0)
        total = 0
        print('Verifying:   0%', end='', file=sys.stderr)
        while total < file_size:
            block_size = min(file_size - total, 1024)
            file_data = f.read(block_size)
            flash_data = dev.read(block_size)
            assert file_data == flash_data
            total += block_size
            print('\rVerifying: %3d%%' % (100*total // file_size), end='', file=sys.stderr)
        print('\rVerifying: 100%', file=sys.stderr)

    if flash:
        dev.write('L')
        response = dev.read(1)
        assert response == 'D'

    dev.close()
    f.close()

parser = argparse.ArgumentParser(description='Upload .bit files to the Mojo Board.')
parser.add_argument('-d', metavar='device', default='/dev/ttyACM0', help='Mojo serial device (default=/dev/ttyACM0)')
parser.add_argument('-s', metavar='bitfile', help='Write bitfile to sram')
parser.add_argument('-f', metavar='bitfile', help='Write bitfile to flash')
parser.add_argument('-v', action='store_true', help='Verify after writing (flash only)')
parser.add_argument('-e', action='store_true', help='Erase flash')
args = parser.parse_args()

if args.e:
    assert not args.s
    assert not args.f
    assert not args.v
    clear_flash(args.d)

elif args.f != None:
    assert not args.s
    assert not args.e
    send_binary(args.d, args.f, True, args.v)

elif args.s != None:
    assert not args.f
    assert not args.v
    assert not args.e
    send_binary(args.d, args.s, False, False)

else:
    parser.print_help()

