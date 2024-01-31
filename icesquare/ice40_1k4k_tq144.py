#!/usr/bin/env python3

import re

print("""EESchema-LIBRARY Version 2.3
#encoding utf-8
#
# iCE40_1K4K_TQ144
#
DEF iCE40_1K4K_TQ144 U 0 40 Y Y 6 F N
F0 "U" -200 250 60 H V C CNN
F1 "iCE40_1K4K_TQ144" -200 150 60 H V C CNN
F2 "" 0 0 60 H V C CNN
F3 "" 0 0 60 H V C CNN
DRAW""")

pin_y_cursors = 6 * [0]
special_pins = set()
nc_pins = set()

def add_pin(name, pin, unit, etype):
    print("X %s %d 200 %d 200 L 50 50 %d 1 %s" % (name, pin, pin_y_cursors[unit], unit+1, etype));
    pin_y_cursors[unit] -= 100

def rename_func(func):
    match = re.match(r"^.*_(SDO|SDI|SCK|SS)$", func)
    if match: return match.group(1)

    # match = re.match(r"^(.*)_(GBIN[0123457]|CBSEL[01])$", func)
    # if match: return match.group(1)

    return func

with open("datasheets/ice40/iCE40144-pinTQFPMigration.csv", "r") as f:
    for line in f:
        line = line.strip().split("\t")
        assert len(line) == 9

        func_1k, pin_1k, type_1k, bank_1k, _, \
                func_4k, pin_4k, type_4k, bank_4k = line

        func_1k = rename_func(func_1k)
        func_4k = rename_func(func_4k)

        assert pin_1k == pin_4k
        pin = int(pin_1k)

        if type_1k in ["DPIO", "GBIN"]: type_1k = "PIO"
        if type_4k in ["DPIO", "GBIN"]: type_4k = "PIO"

        if (func_1k.find("_GBIN") >= 0 or func_1k.find("_CBSEL") >= 0 or
                func_4k.find("_GBIN") >= 0 or func_4k.find("_CBSEL") >= 0) and \
                not func_1k.endswith("_GBIN6"):
            nc_pins.add(pin)
            continue

        if type_1k == "PIO" and type_4k == "PIO":
            assert func_1k[0:4] == func_4k[0:4]
            assert bank_1k == bank_4k
            if func_1k.endswith("_GBIN6"):
                special_pins.add(("GBIN6", pin))
            else:
                add_pin(func_1k + func_4k[3:], pin, int(bank_1k) + 1, "U")
            continue

        if type_1k in ["PIO", "NC"] and type_4k in ["PIO", "NC"]:
            nc_pins.add(pin)
            continue

        if type_1k in ["VCCIO", "GND", "VCC", "VPP", "GNDPLL", "VCCPLL", "CONFIG", "SPI", "NC"] and \
                type_4k in ["VCCIO", "GND", "VCC", "VPP", "GNDPLL", "VCCPLL", "CONFIG", "SPI", "NC"]:
            if type_1k == "NC":
                func_1k, pin_1k, type_1k, bank_1k = func_4k, pin_4k, type_4k, bank_4k
            if type_4k == "NC":
                func_4k, pin_4k, type_4k, bank_4k = func_1k, pin_1k, type_1k, bank_1k

            assert type_1k == type_4k and func_1k == func_4k and bank_1k == bank_4k
            special_pins.add((func_1k, pin))
            continue

        assert False

for i, p in enumerate(sorted(nc_pins)):
    add_pin("NC_%d" % i, p, 5, "N")

for n, p in sorted(special_pins):
    add_pin(n, p, 0, "U")

for i in range(len(pin_y_cursors)):
    print("S -600 100 0 %d %d 1 0 f" % (pin_y_cursors[i], i+1))

print("""ENDDRAW
ENDDEF
#
#End Library""")

