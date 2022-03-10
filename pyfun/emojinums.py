#!/usr/bin/env python3

def intToHearts(n):
    return ("🥺" * n).replace("🥺🥺🥺", "💕").replace("💕💕💕", "💋")

def heartsToInt(s):
    return sum([{"🥺": 1, "💕": 3, "💋": 9}[c] for c in s])

if __name__ == '__main__':
    for i in range(1, 27):
        a = intToHearts(i)
        b = heartsToInt(a)
        print(f"{i:2d} {b:2d} {a}")
        assert i == b
