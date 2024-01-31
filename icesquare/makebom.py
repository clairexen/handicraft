#!/usr/bin/env python3

import xml.etree.ElementTree as ET

tree = ET.parse("icesquare.xml")
parts = dict()

prices = {
    "490-1532-1-ND": 0.02570,
    "490-10474-1-ND": 0.05990,
    "511-1292-1-ND": 0.26120,
    "S5559-ND": 1.11000,
    "P100BYCT-ND": 0.04920,
    "P10KBYCT-ND": 0.04920,
    "P12KBYCT-ND": 0.04920,
    "P1.0KBYCT-ND": 0.04920,
    "P2.2KBYCT-ND": 0.04920,
    "768-1101-1-ND": 3.88000,
    "220-1572-ND": 6.37000,
    "535-11724-1-ND": 1.12000,
    "641-1332-1-ND": 0.43000,
    "609-4618-1-ND": 0.42000,
    "576-3880-5-ND": 0.46000,
    "1274-1134-ND": 0.57000,
    "220-1565-ND": 5.22000,
}

for comp in tree.findall(".//comp"):
    ref = comp.attrib.get("ref")
    value = comp.find("./value").text
    footprint = comp.find("./footprint").text
    digikey = comp.find("./fields/field[@name='DigiKey']")

    if digikey is None:
        digikey = "NO_DIGIKEY_ID"
        if value.startswith("TEST_"): continue
        if footprint == "Footprints:MOUNTINGHOLE": continue
        if footprint == "Resistors_SMD:R_0603" and value == "100": digikey = "P100BYCT-ND"
        if footprint == "Resistors_SMD:R_0603" and value == "10k": digikey = "P10KBYCT-ND"
        if footprint == "Resistors_SMD:R_0603" and value == "12k": digikey = "P12KBYCT-ND"
        if footprint == "Resistors_SMD:R_0603" and value == "2k2": digikey = "P2.2KBYCT-ND"
        if footprint == "Resistors_SMD:R_0603" and value == "1k": digikey = "P1.0KBYCT-ND"
        if footprint == "Capacitors_SMD:C_0603" and value == "0.1uF": digikey = "490-1532-1-ND"
        if footprint == "Capacitors_SMD:C_0603" and value == "10uF": digikey = "490-10474-1-ND"
    else:
        digikey = digikey.text

    idx = "%s %s %s %s" % (ref[0], value, footprint, digikey)
    if idx not in parts:
        parts[idx] = {
            "refchar": ref[0],
            "value": value,
            "footprint": footprint,
            "digikey": digikey,
            "refs": set()
        }
    parts[idx]["refs"].add(ref)

with open("icesquare.bom", "w") as f:
    total_price = 0
    for _, p in sorted(parts.items()):
        price = len(p["refs"]) * prices[p["digikey"]] if p["digikey"] in prices else 0.0
        print("%s %-20s %2dx %-20s %7.2f EUR  %s" % (p["refchar"], p["value"],
                len(p["refs"]), p["digikey"], price, " ".join(sorted(p["refs"]))), file=f)
        total_price += price
    print("Total: %.2f EUR" % total_price, file = f)

