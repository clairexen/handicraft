#!/usr/bin/python3

import sys

spi_ctrl_core = "spi_ctrl"
pmod_pins = dict()
ctrl_pins = dict()

core_db = {
    "servo_pwm": {
        "min_pmods": 1,
        "max_pmods": 32,
        "num_endpoints": 2
    },
    "echo_test": {
        "min_pmods": 0,
        "max_pmods": 0,
        "num_endpoints": 1
    },
}

endpoints_db = list()

text_design_ports = list()
text_internal_wires = list()
text_module_instances = list()
text_placement_constr = list()
text_code = list()

text_design_ports.append("input clk")
text_internal_wires.append("wire resetn")
text_code.append("reg [3:0] resetn_count = 0;")
text_code.append("assign resetn = &resetn_count;")
text_code.append("always @(posedge clk) if (!resetn) resetn_count <= resetn_count + 1;")

text_design_ports.append("input spi_sclk")
text_design_ports.append("input spi_mosi")
text_design_ports.append("output spi_miso")
text_design_ports.append("input [7:0] spi_csel")

text_internal_wires.append("wire spi_ctrl_si")
text_internal_wires.append("wire spi_ctrl_so")
text_internal_wires.append("wire spi_ctrl_hd")
text_internal_wires.append("wire [7:0] spi_ctrl_di")
all_spi_ctrl_dout = list()

with open("config.txt", "r") as cfg_file:
    config = [line.split() for line in cfg_file if len(line.split()) > 0]

cursor = 0
while cursor < len(config):
    if config[cursor][0] == "OPTION":
        if config[cursor][1] == "board" and config[cursor][2] == "hx8kdev":

            pmod_pins["A"] = "T5 T6  T7  T9  R5 R6  T8  R9 ".split()
            pmod_pins["B"] = "E2 F2  G3  H2  F1 G1  H1  D2 ".split()
            pmod_pins["C"] = "P8 T10 T11 N10 P9 R10 P10 M11".split()
            pmod_pins["D"] = "J2 K3  L3  M2  J1 K1  L1  M1 ".split()

            ctrl_pins["clk"] = "J3"
            ctrl_pins["spi_sclk"] = "C16"
            ctrl_pins["spi_mosi"] = "D16"
            ctrl_pins["spi_miso"] = "E16"
            ctrl_pins["spi_csel[0]"] = "G14"
            ctrl_pins["spi_csel[1]"] = "K14"
            ctrl_pins["spi_csel[2]"] = "K15"
            ctrl_pins["spi_csel[3]"] = "M16"
            ctrl_pins["spi_csel[4]"] = "F14"
            ctrl_pins["spi_csel[5]"] = "J14"
            ctrl_pins["spi_csel[6]"] = "K16"
            ctrl_pins["spi_csel[7]"] = "L16"

            cursor += 1
            continue

        if config[cursor][1] == "spi_type":
            spi_ctrl_core = "spi_ctrl_" + config[cursor][2]
            cursor += 1
            continue

        raise Exception("Invalid option or value in line %d of config.txt!" % (cursor+1))

    if config[cursor][0] == "CORE":
        core_type = config[cursor][1]
        core_name = config[cursor][2]
        pmods = list()
        cursor += 1

        while cursor < len(config):
            if config[cursor][0] == "PMODS":
                pmods += config[cursor][1:]
                cursor += 1
                continue
            break

        assert core_type in core_db
        assert core_db[core_type]["min_pmods"] <= len(pmods) <= core_db[core_type]["max_pmods"]

        pmod_i = list()
        pmod_o = list()
        pmod_d = list()

        for pm_index, pm in enumerate(pmods):
            text_design_ports.append("inout [7:0] pmod_%s_%d_%s" % (core_name, pm_index, pm))
            text_internal_wires.append("wire [7:0] pmod_%s_%d_%s_i" % (core_name, pm_index, pm))
            text_internal_wires.append("wire [7:0] pmod_%s_%d_%s_o" % (core_name, pm_index, pm))
            text_internal_wires.append("wire [7:0] pmod_%s_%d_%s_d" % (core_name, pm_index, pm))
            text_module_instances.append("SB_IO #(")
            text_module_instances.append("  .PIN_TYPE(6'b1010_01),")
            text_module_instances.append("  .PULLUP(1'b0),")
            text_module_instances.append("  .NEG_TRIGGER(1'b0),")
            text_module_instances.append("  .IO_STANDARD(\"SB_LVCMOS\")")
            text_module_instances.append(") pmod_%s_%d_%s_io [7:0] (" % (core_name, pm_index, pm))
            text_module_instances.append("  .PACKAGE_PIN(pmod_%s_%d_%s)," % (core_name, pm_index, pm))
            text_module_instances.append("  .LATCH_INPUT_VALUE(1'b0),")
            text_module_instances.append("  .CLOCK_ENABLE(1'b0),")
            text_module_instances.append("  .INPUT_CLK(1'b0),")
            text_module_instances.append("  .OUTPUT_CLK(1'b0),")
            text_module_instances.append("  .OUTPUT_ENABLE(pmod_%s_%d_%s_d)," % (core_name, pm_index, pm))
            text_module_instances.append("  .D_OUT_0(pmod_%s_%d_%s_o)," % (core_name, pm_index, pm))
            text_module_instances.append("  .D_IN_0(pmod_%s_%d_%s_i)" % (core_name, pm_index, pm))
            text_module_instances.append(");")
            for idx, pin in enumerate(pmod_pins[pm]):
                text_placement_constr.append("set_io pmod_%s_%d_%s[%d] %s" % (core_name, pm_index, pm, idx, pin))
            pmod_i.append("pmod_%s_%d_%s_i" % (core_name, pm_index, pm))
            pmod_o.append("pmod_%s_%d_%s_o" % (core_name, pm_index, pm))
            pmod_d.append("pmod_%s_%d_%s_d" % (core_name, pm_index, pm))

        text_internal_wires.append("wire [7:0] dout_%s" % core_name)
        all_spi_ctrl_dout.append("dout_%s" % core_name)

        epsel = list()
        for i in range(core_db[core_type]["num_endpoints"]):
            n = "epsel_%s_%d" % (core_name, i)
            text_internal_wires.append("wire %s" % n)
            endpoints_db.append([n, core_name, i])
            epsel.append(n)

        text_module_instances.append("ico_%s #(" % core_type)
        text_module_instances.append("  .NUM_PMODS(%d)" % len(pmods))
        text_module_instances.append(") c_%s (" % core_name)
        text_module_instances.append("  .clk(clk),")
        text_module_instances.append("  .resetn(resetn),")
        text_module_instances.append("  .spi_ctrl_si(spi_ctrl_si),")
        text_module_instances.append("  .spi_ctrl_so(spi_ctrl_so),")
        text_module_instances.append("  .spi_ctrl_hd(spi_ctrl_hd),")
        text_module_instances.append("  .spi_ctrl_di(spi_ctrl_di),")
        text_module_instances.append("  .spi_ctrl_do(dout_%s)," % core_name)
        if len(pmods):
            text_module_instances.append("  .epsel({%s})," % (", ".join(reversed(epsel))))
            text_module_instances.append("  .pmod_i({%s})," % (", ".join(reversed(pmod_i))))
            text_module_instances.append("  .pmod_o({%s})," % (", ".join(reversed(pmod_o))))
            text_module_instances.append("  .pmod_d({%s})" % (", ".join(reversed(pmod_d))))
        else:
            text_module_instances.append("  .epsel({%s})" % (", ".join(reversed(epsel))))
        text_module_instances.append(");")

        continue

    raise Exception("Parser error in line %d of config.txt!" % (cursor+1))

text_module_instances.append("%s #(" % spi_ctrl_core)
text_module_instances.append("    .NUM_ENDPOINTS(%d)" % len(endpoints_db))
text_module_instances.append(") spi_ctrl (")
text_module_instances.append("    .clk(clk),")
text_module_instances.append("    .resetn(resetn),")
text_module_instances.append("    .spi_sclk(spi_sclk),")
text_module_instances.append("    .spi_mosi(spi_mosi),")
text_module_instances.append("    .spi_miso(spi_miso),")
text_module_instances.append("    .spi_csel(spi_csel),")
text_module_instances.append("    .spi_ctrl_si(spi_ctrl_si),")
text_module_instances.append("    .spi_ctrl_so(spi_ctrl_so),")
text_module_instances.append("    .spi_ctrl_hd(spi_ctrl_hd),")
text_module_instances.append("    .spi_ctrl_di(spi_ctrl_di),")
text_module_instances.append("    .spi_ctrl_do(%s)," % (" | ".join(all_spi_ctrl_dout)))
text_module_instances.append("    .epsel({%s})" % (", ".join([t[0] for t in reversed(endpoints_db)])))
text_module_instances.append(");")

for net, pin in ctrl_pins.items():
    text_placement_constr.append("set_io %s %s" % (net, pin))

with open("chip.v", "w") as f:
    print("`timescale 1 ns / 1 ps", file=f)
    print("module chip (", file=f)
    print("  %s" % ",\n  ".join(text_design_ports), file=f)
    print(");", file=f)
    for line in text_internal_wires:
        print("  %s;" % line, file=f)
    for line in text_module_instances:
        print("  %s" % line, file=f)
    for line in text_code:
        print("  %s" % line, file=f)
    print("endmodule", file=f)

with open("chip.pcf", "w") as f:
    for line in text_placement_constr:
        print(line, file=f)



