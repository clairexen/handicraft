#!/usr/bin/env python3

bramdata = list()

hram_ck = 1
hram_cs = 1
hram_rwds_dir = 0
hram_rwds_dout = 0
hram_dq_dir = 0
hram_dq_dout = 0

def step():
    v = hram_dq_dout & 255
    v = v | ((hram_dq_dir & 1) << 8)
    v = v | ((hram_rwds_dout & 1) << 9)
    v = v | ((hram_rwds_dir & 1) << 10)
    v = v | ((hram_cs & 1) << 11)
    v = v | ((hram_ck & 1) << 12)
    bramdata.append(v)


def make_command_addr(rw, addrspace, bursttype, rowaddr, coladdr):
    bytelist = list()
    bytelist.append(coladdr & 7)
    bytelist.append(0)
    bytelist.append(((coladdr & 255) >> 3) | ((rowaddr & 3) << 6))
    bytelist.append((rowaddr >> 2) & 255)
    bytelist.append((rowaddr >> 10) & 255)
    bytelist.append(((rw & 1) << 7) | ((addrspace & 1) << 6) | ((bursttype & 1) << 5))
    return bytelist

def idle_state():
    global hram_ck, hram_cs
    global hram_rwds_dir, hram_rwds_dout
    global hram_dq_dir, hram_dq_dout

    hram_ck = 0
    hram_cs = 1
    hram_rwds_dir = 0
    hram_dq_dir = 0

    for i in range(32):
        step()

def read_register(addr):
    global hram_ck, hram_cs
    global hram_rwds_dir, hram_rwds_dout
    global hram_dq_dir, hram_dq_dout

    idle_state()
    hram_cs = 0
    step()

    coladdr = addr & 1023
    rowaddr = addr >> 10

    for v in reversed(make_command_addr(1, 1, 1, rowaddr, coladdr)):
        hram_dq_dir = 1
        hram_dq_dout = v
        step()
        hram_ck = ~hram_ck
        step()

    hram_dq_dir = 0

    for i in range(24):
        step()
        hram_ck = ~hram_ck
        step()

def write_memory(addr, words):
    global hram_ck, hram_cs
    global hram_rwds_dir, hram_rwds_dout
    global hram_dq_dir, hram_dq_dout

    idle_state()
    hram_cs = 0
    step()

    coladdr = addr & 1023
    rowaddr = addr >> 10

    for v in reversed(make_command_addr(0, 0, 1, rowaddr, coladdr)):
        hram_dq_dir = 1
        hram_dq_dout = v
        step()
        hram_ck = ~hram_ck
        step()

    hram_dq_dir = 1
    hram_dq_dout = 0xff

    for i in range(22):
        step()
        hram_ck = ~hram_ck
        step()

    hram_rwds_dir = 1
    hram_rwds_dout = 0

    for v in words:
        hram_dq_dout = v >> 8
        step()
        hram_ck = ~hram_ck
        step()

        hram_dq_dout = v & 255
        step()
        hram_ck = ~hram_ck
        step()

def read_memory(addr, words):
    global hram_ck, hram_cs
    global hram_rwds_dir, hram_rwds_dout
    global hram_dq_dir, hram_dq_dout

    idle_state()
    hram_cs = 0
    step()

    coladdr = addr & 1023
    rowaddr = addr >> 10

    for v in reversed(make_command_addr(1, 0, 1, rowaddr, coladdr)):
        hram_dq_dir = 1
        hram_dq_dout = v
        step()
        hram_ck = ~hram_ck
        step()

    hram_dq_dir = 0

    for i in range(22 + 2*words):
        step()
        hram_ck = ~hram_ck
        step()


## Run Tests

read_register(0x00000000)
read_register(0x00000001)

write_memory(0x0000, list(range(0x20)))
read_memory(0x0010, 0x10)


## Fill rest of the buffer

hram_ck = 0
hram_cs = 1
hram_rwds_dir = 0
hram_dq_dir = 0

while len(bramdata) < 1024:
    step()


## Write HEX File

with open("libretto_data.hex", "w") as f:
    for v in bramdata:
        print("%04x" % v, file=f)

