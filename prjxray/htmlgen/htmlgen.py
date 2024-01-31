#!/usr/bin/env python3

import os, json, re
from collections import defaultdict

print("Reading tile data from ../database/tilegrid.json.")

with open("../database/tilegrid.json", "r") as f:
    tiledata = json.load(f)

site_to_tile = dict()

for tile, props in tiledata.items():
    for site in props["SITES"].keys():
        site_to_tile[site] = tile

cbits_by_position = dict()
cbits_by_path = dict()
cbits_by_tile = dict()

tile_to_frame = defaultdict(set)
frame_to_tile = defaultdict(set)

for fn in os.listdir("../database"):
    if not fn.endswith(".cbits"):
        continue

    fn = "../database/%s" % fn
    print("Reading config bit data from %s." % fn)

    with open(fn, "r") as f:
        for line in f:
            frame, bit, path = line.split()
            path = path.split(".")

            if frame not in cbits_by_position:
                cbits_by_position[frame] = dict()

            cbits_by_position[frame][bit] = tuple(path)

            cursor = cbits_by_path
            for path_element in path[:-1]:
                if path_element not in cursor:
                    cursor[path_element] = dict()
                cursor = cursor[path_element]

            cursor[path[-1]] = (frame, bit)

for frame in sorted(cbits_by_position.keys()):
    print("Writing frame_%s.html." % frame)

    with open("frame_%s.html" % frame, "w") as f:
        print("<h3>Project X-Ray: %s configuration frame %s</h3>" % (os.environ['XRAY_PART'], frame), file=f)

        print("<table style=\"white-space: pre; font-family: monospace;\" border>", file=f)
        print("<tr><th>Word #</th><th>Bit #</th><th>Bit-ID</th><th>Tile</th><th>Function</th></tr>", file=f)

        table_data = list()

        for word_idx in range(101):
            for bit_idx in range(32):
                bitname = "%c%c%02d" % (chr(ord('A') + (word_idx // 26)), chr(ord('A') + (word_idx % 26)), bit_idx)
                tile, func = "----", "----"

                if word_idx == 50:
                    func = "FRAME ECC"

                if bitname in cbits_by_position[frame]:
                    path = cbits_by_position[frame][bitname]
                    func = ".".join(path)
                    if path[0] in site_to_tile:
                        tile = site_to_tile[path[0]]

                table_data.append(("%d" % word_idx, "%d" % bit_idx, bitname, tile, func))

        compressed_table_data = list()

        group_begin = 0
        while group_begin < len(table_data):
            group_end = group_begin
            while group_end+1 < len(table_data):
                k = group_end+1

                # do not merge different words
                if table_data[group_begin][0] != table_data[k][0]:
                    break

                # do not merge different tiles
                if table_data[group_begin][3] != table_data[k][3]:
                    break

                f1 = re.sub(r"\[[0-9]*\]", "[]", table_data[group_begin][4])
                f2 = re.sub(r"\[[0-9]*\]", "[]", table_data[k][4])

                if f1 != f2:
                    break

                group_end = k

            if group_begin == group_end:
                compressed_table_data.append(table_data[group_begin])

            else:
                word_idx = table_data[group_begin][0]
                bit_idx = "%2s .. %2s" % (table_data[group_begin][1], table_data[group_end][1])
                bitname = "%s .. %s" % (table_data[group_begin][2], table_data[group_end][2])
                tile = table_data[group_begin][3]

                flist = list()
                for i in range(group_begin, group_end+1):
                    flist.append(re.sub(r"(.*\[|\].*)", "", table_data[i][4]))

                func = re.sub(r"\[[0-9]*\]", "[%s]" % (",".join(flist)), table_data[group_begin][4])
                compressed_table_data.append((word_idx, bit_idx, bitname, tile, func))

            group_begin = group_end+1

        for word_idx, bit_idx, bitname, tile, func in compressed_table_data:
            if tile != "----":
                if tile not in cbits_by_tile:
                    cbits_by_tile[tile] = list()
                cbits_by_tile[tile].append((frame, word_idx, bit_idx, bitname, func))

                tile_to_frame[tile].add(frame)
                frame_to_tile[frame].add(tile)

                tile = "<a href=\"tile_%s.html\">%s</a>" % (tile, tile)

            print("<tr><td align=\"right\">%s</td><td align=\"right\">%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (word_idx, bit_idx, bitname, tile, func), file=f)

        print("</table>", file=f)

for tile in sorted(cbits_by_tile.keys()):
    print("Writing tile_%s.html." % tile)

    with open("tile_%s.html" % tile, "w") as f:
        print("<h3>Project X-Ray: %s tile %s</h3>" % (os.environ['XRAY_PART'], tile), file=f)

        print("<h4>Configuration Column</h4>", file=f)

        column = set()
        for frame in tile_to_frame[tile]:
            column |= frame_to_tile[frame]

        print("<div style=\"-webkit-column-count: 3; -moz-column-count: 3; column-count: 3; max-width: 800px;\"><ul style=\"margin: 0;\">", file=f)
        for t in sorted(column):
            if t == tile:
                print("<li>%s</li>" % t, file=f)
            else:
                print("<li><a href=\"tile_%s.html\">%s</a></li>" % (t, t), file=f)
        print("</ul></div>", file=f)

        print("<h4>Configuration Bits</h4>", file=f)

        print("<table style=\"white-space: pre; font-family: monospace;\" border>", file=f)
        print("<tr><th>Frame</th><th>Word #</th><th>Bit #</th><th>Bit-ID</th><th>Function</th></tr>", file=f)

        for frame, word_idx, bit_idx, bitname, func in cbits_by_tile[tile]:
            frame = "<a href=\"frame_%s.html\">%s</a>" % (frame, frame)
            print("<tr><td>%s</td><td align=\"right\">%s</td><td align=\"right\">%s</td><td>%s</td><td>%s</td></tr>" % (frame, word_idx, bit_idx, bitname, func), file=f)

        print("</table>", file=f)

