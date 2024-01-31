#!/usr/bin/env python3

import os, json

with open("../../database/tilegrid.json", "r") as f:
    tiledata = json.load(f)

num_x = 0
num_y = 0
tile_by_gridpos = dict()

for tile, props in tiledata.items():
    x = int(props["GRID_POINT_X"])
    y = int(props["GRID_POINT_Y"])

    num_x = max(num_x, x+1)
    num_y = max(num_y, y+1)

    xy = (x, y)
    assert xy not in tile_by_gridpos
    tile_by_gridpos[xy] = tile

if False:
    for pos, tile in sorted(tile_by_gridpos.items()):
        print("%5d %5d %s" % (pos[0], pos[1], tile))

print("<h3>Project X-Ray: %s tile grid (total size: %d x %d, fuzzing region: %s)</h3>" % (os.environ['XRAY_PART'], num_x, num_y, os.environ['XRAY_REGION']))

print("<table border style=\"width:%dpx\">" % (num_x * 10))

for y in range(num_y):
    print("<tr>")

    for x in range(num_x):
        xy = (x, y)
        assert xy in tile_by_gridpos

        tile = tile_by_gridpos[xy]
        assert tile in tiledata

        color = "#fff"
        if tile.startswith("NULL_"): color = "#000"

        if tile.startswith("INT_"): color = "#00f"
        if tile.startswith("INT_INTERFACE_"): color = "#44a"
        if tile.startswith("INT_FEEDTHRU_"): color = "#888"
        if tile.startswith("T_TERM_INT_"): color = "#44a"
        if tile.startswith("B_TERM_INT_"): color = "#44a"

        if tile.startswith("CLBLL_"): color = "#f80"
        if tile.startswith("CLBLM_"): color = "#ff0"

        if tile.startswith("BRAM_"): color = "#f00"
        if tile.startswith("BRAM_INT_"): color = "#808"
        if tile.startswith("DSP_"): color = "#f00"

        if tile.startswith("PCIE_"): color = "#4a4"

        if tiledata[tile]["FUZZ"] == "0":
            r = 3 + int(color[1], 16) // 3
            g = 3 + int(color[2], 16) // 3
            b = 3 + int(color[3], 16) // 3
            color = "#%x%x%x" % (r, g, b)

        sites = ""
        for site, props in sorted(tiledata[tile]["SITES"].items()):
            if sites == "": sites += "\n"
            sites += "\n%s (%s%s)" % (site, props["SITE_TYPE"], (", " + props["CLOCK_REGION"]) if "CLOCK_REGION" in props else "")

        print("<td title=\"%s\nGRID_XY=%d,%d%s\" style=\"width:10px; height:10px; background-color:%s\"> </td>" % (tile, x, y, sites, color))

    print("</tr>")

print("</table>")

