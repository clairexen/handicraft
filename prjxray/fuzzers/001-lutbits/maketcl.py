#!/usr/bin/env python3

import os, json, random

lutlist = dict()

print("Reading tile data..")

with open("../../database/tilegrid.json", "r") as f:
    tiledata = json.load(f)

print("Generating test design..")

bus_width = 8
input_ports = ["i%d" % i for i in range(bus_width)]
output_ports = ["o%d" % i for i in range(bus_width)]

with open("top.v", "w") as f, open("place.tcl", "w") as p:
    print("module top (input %s, output %s);" % (", ".join(input_ports), ", ".join(output_ports)), file=f)
    netcache = input_ports[:]
    netcacheidx = 0
    netindex = 0

    for tile, props in sorted(tiledata.items()):
        if props["TILE_TYPE"] not in ["CLBLL_L", "CLBLL_R", "CLBLM_L", "CLBLM_R"]:
            continue

        if props["FUZZ"] == "0":
            continue

        newnets = list()

        for site, site_props in sorted(props["SITES"].items()):
            site_type = site_props["SITE_TYPE"]

            if site_type not in ["SLICEL", "SLICEM"]:
                continue

            for lut in ["A6LUT", "B6LUT", "C6LUT", "D6LUT"]:
                print("  wire n%d;" % netindex, file=f)
                print("  LUT6 #(", file=f)
                print("    .INIT(64'h0000000000000001)", file=f)
                print("  ) %s_%s (" % (site, lut), file=f)
                for i in range(6):
                    print("    .I%d(%s)," % (i, netcache[netcacheidx]), file=f)
                    netcacheidx = (netcacheidx + 1) % bus_width
                print("    .O(n%d)" % netindex, file=f)
                print("  );", file=f)

                print("set_property -dict {IS_LOC_FIXED 1 IS_BEL_FIXED 1 LOC %s BEL %s.%s} [get_cells %s_%s]" % (site, site_type, lut, site, lut), file=p)

                lutlist["%s_%s" % (site, lut)] = "%s.%s" % (site, lut)
                newnets.append("n%d" % netindex)
                netindex += 1

        if len(newnets) > 0:
            assert len(newnets) <= bus_width

            random.shuffle(netcache)
            netcache[0:len(newnets)] = newnets

    print("  assign {%s} = {%s};" % (", ".join(output_ports), ", ".join(netcache)), file=f)
    print("endmodule", file=f)

with open("top.xdc", "w") as f:
    pins = "F21 G22 G21 D21 E21 D22 E22 A21 B21 B22 C22 C20 D20 F20 F19 A19 A18 A20 B20 E18 F18 D19 E19 C19".split()
    ports = input_ports + output_ports

    for pin, port in zip(pins, ports):
        print("set_property -dict {PACKAGE_PIN %s IOSTANDARD LVCMOS33} [get_ports %s]" % (pin, port), file=f)

    print("set_property LOCK_PINS {I0:A1 I1:A2 I2:A3 I3:A4 I4:A5 I5:A6} [get_cells -filter {REF_NAME == LUT6}]", file=f)

print("Generated design contains %d LUTs." % len(lutlist))

with open("makebits.tcl", "w") as f:
    print("create_project -part %s -force vivadoprj vivadoprj" % os.environ['XRAY_PART'], file=f)
    print("read_verilog top.v", file=f)
    print("read_xdc top.xdc", file=f)
    print("synth_design -top top", file=f)
    print("source -notrace place.tcl", file=f)
    print("place_design", file=f)
    print("route_design", file=f)

    print("set_property CFGBVS VCCO [current_design]", file=f)
    print("set_property CONFIG_VOLTAGE 3.3 [current_design]", file=f)
    print("set_property BITSTREAM.GENERAL.PERFRAMECRC YES [current_design]", file=f)

    print("write_checkpoint -force base.dcp", file=f)

    for i in range(50):
        prefix = "test%03d" % i

        print("Generating %s.." % prefix)

        with open(prefix + ".tags", "w") as tags:
            with open(prefix + ".tcl", "w") as tcl:
                for lut, bit_prefix in lutlist.items():
                    while True:
                        inithex = "".join([random.choice('0123456789ABCDEF') for x in range(16)])
                        initval = int(inithex, 16)
                        initcarry = False

                        for k in range(64):
                            if (initval & (1 << k)) != 0:
                                initcarry = not initcarry

                        if initcarry: break

                    print("set_property INIT 64'h%s [get_cells %s]" % (inithex, lut), file=tcl)

                    for k in range(64):
                        if (initval & (1 << k)) != 0:
                            print("%s.INIT[%d]" % (bit_prefix, k), file=tags)

        print("source -notrace %s.tcl" % prefix, file=f)
        print("write_bitstream -force %s.bit" % prefix, file=f)

