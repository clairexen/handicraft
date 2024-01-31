#!/usr/bin/env python3

def sgfread(filename):
    sgftokens = []
    with open(filename, "r") as f:
        cur_buffer = ""
        cur_parseval = False
        for line in f:
            for c in line:
                if not cur_parseval:
                    if c in [' ', '\r', '\n', '\t']:
                        continue;
                    if c in ['(', ';', ')']:
                        if len(cur_buffer):
                            sgftokens.append(cur_buffer)
                            cur_buffer = ""
                        sgftokens.append(c)
                        continue
                cur_buffer += c
                if c == '[':
                    cur_parseval = True
                if c == ']':
                    cur_parseval = False
                    sgftokens.append(cur_buffer)
                    cur_buffer = ""

    stack = []
    sgfdata = None
    for token in sgftokens:
        if token == '(':
            newgroup = []
            if len(stack):
                stack[-1].append(("group", newgroup))
            else:
                sgfdata = newgroup
            stack.append(newgroup)
        elif token == ')':
            stack.pop()
        elif token == ';':
            stack[-1].append(("node", []))
        else:
            prop_name, prop_value = token.split('[')
            prop_value = prop_value.rstrip(']')
            stack[-1][-1][1].append(("prop", prop_name, prop_value))

    return sgfdata

def sgfpositions(sgfdata, positions):
    def worker(pos, out_positions, sgfgroup):
        for node in sgfgroup:
            if node[0] == "group":
                pos = worker(pos, out_positions, node[1])
                continue

            assert node[0] == "node"

            for prop in node[1]:
                assert prop[0] == "prop"

                if prop[1] == "SZ":
                    sz = int(prop[2])
                    pos = [[" "] * sz for i in range(sz)]
                    positions.add(tuple(["".join(line) for line in pos]))

                if prop[1] in ["W", "B"] and len(prop[2]) == 2:
                    x = ord(prop[2][0]) - ord('a')
                    y = ord(prop[2][1]) - ord('a')
                    pos[x][y] = prop[1]
                    positions.add(tuple(["".join(line) for line in pos]))

        return pos

    pos = worker(None, positions, sgfdata)
    return None if pos is None else tuple(["".join(line) for line in pos])

def sgfpermpos(position):
    permpositions = set()
    sz = len(position)
    for flip_x in [False, True]:
        for flip_y in [False, True]:
            for transpose in [False, True]:
                ppos = [[" "] * sz for i in range(sz)]
                for x in range(sz):
                    for y in range(sz):
                        u = sz-x-1 if flip_x else x
                        v = sz-y-1 if flip_y else y
                        if transpose: u, v = v, u
                        ppos[x][y] = position[u][v]
                permpositions.add(tuple(["".join(line) for line in ppos]))
    return permpositions

def sgfprettystr(sgfdata, indent=None):
    text = []
    if indent is None:
        text.append("`- root\n")
        text.append(sgfprettystr(sgfdata, "  "))
    else:
        for el in sgfdata:
            if el[0] in ["node", "group"]:
                text.append("%s`- %s\n" % (indent, el[0]))
                text.append(sgfprettystr(el[1], indent + "  "))
            else:
                text.append("%s`- %s\n" % (indent, " ".join(el)))
    return "".join(text)

