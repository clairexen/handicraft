#!/usr/bin/env python

test_input = []
#test_input = list(reversed("by bx ay bz ? yc".split()))

player = "X"

# initialize state matrix
state = {}
for row in range(3):
    for col in range(3):
        state[row,col] = " "

# create list of all lines
lines = []
for row in range(3):
    lines.append(((row, 0), (row, 1), (row, 2)))
for col in range(3):
    lines.append(((0, col), (1, col), (2, col)))
lines.append(((0, 0), (1, 1), (2, 2)))
lines.append(((2, 0), (1, 1), (0, 2)))

def print_state():
    print()
    print("     A   B   C")
    print("   +---+---+---+")
    for row in range(3):
        print(" %c | %c | %c | %c |" % (chr(ord("X")+row), state[row,0], state[row,1], state[row,2]))
        print("   +---+---+---+")
    print()

def print_analysis():
    print()
    print("   +" + ("---+" * len(lines)))
    for row in range(3):
        s = "   |"
        for line in lines:
            for col in range(3):
                s += "+" if (row, col) in line else " "
            s += "|"
        print(s)
    print("   +" + ("---+" * len(lines)))
    for row in range(3):
        s = "   |"
        for line in lines:
            for col in range(3):
                c = state[row, col]
                if (row, col) in line:
                    if c == " ": c = "+"
                else:
                    c = c.lower()
                s += c
            s += "|"
        print(s)
    print("   +" + ("---+" * len(lines)))
    s = "    "
    for line in lines:
        xs, os, bs = "", "", ""
        for row, col in line:
            if state[row, col] == "X":
                xs += "X"
            elif state[row, col] == "O":
                os += "O"
            else:
                bs += " "
        if len(bs) != 3:
            if xs == "":
                bs = bs.replace(" ", "o")
            elif os == "":
                bs = bs.replace(" ", "x")
            else:
                bs = bs.replace(" ", "-")
        else:
            bs = bs.replace(" ", ".")

        s += xs + bs + os + " "
    print(s)

def get_winner():
    fin_x_won, fin_o_won = False, False
    for line in lines:
        x_won, o_won = True, True
        for row, col in line:
            if state[row, col] != "X":
                x_won = False
            if state[row, col] != "O":
                o_won = False
        if x_won: fin_x_won = True
        if o_won: fin_o_won = True
    assert not (fin_x_won and fin_o_won)
    if fin_x_won:
        return "X"
    if fin_o_won:
        return "O"

    for row in range(3):
        for col in range(3):
            if state[row,col] == " ":
                return None
    return "T"

def tictactoe():
    global player
    while True:
        print_state()
        w = get_winner()

        if w is not None:
            if w in "XO":
                print("Player %c won!" % w)
            else:
                print("It's a tie!")
            return w

        while True:
            prompt = "Player %c, select a free square (e.g. 'AX'), or '?' for analysis, or Q to quit: " % player

            if len(test_input):
                s = test_input.pop()
                print(prompt + s)
            else:
                s = input(prompt).strip().lower()

            if s == "?":
                print_analysis()
                print_state()
                continue

            if s in "qQ":
                return "Q"

            if len(s) != 2:
                continue

            c1, c2 = s
            if c1 in "abc" and c2 in "xyz":
                row = ord(c2)-ord("x")
                col = ord(c1)-ord("a")
            elif c2 in "abc" and c1 in "xyz":
                row = ord(c1)-ord("x")
                col = ord(c2)-ord("a")
            else:
                continue

            if state[row,col] == " ":
                break

        state[row,col] = player
        player = "X" if player == "O" else "O"

tictactoe()
