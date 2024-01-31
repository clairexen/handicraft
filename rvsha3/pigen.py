#!/usr/bin/einv python3
#
# Generate Keccak Rho and Pi steps

rot_table = [
    [ 0, 36,  3, 41, 18], # x=0
    [ 1, 44, 10, 45,  2], # x=1
    [62,  6, 43, 15, 61], # x=2
    [28, 55, 25, 21, 56], # x=3
    [27, 20, 39,  8, 14], # x=4
]

fwd_map = dict()
bwd_map = dict()

for x in range(5):
    for y in range(5):
        src = (x,y)
        dst = (y, (2*x + 3*y) % 5)
        fwd_map[src] = dst
        bwd_map[dst] = src

for src, dst in sorted(fwd_map.items()):
    print("#define B%d%d A%d%d" % (src[0], src[1], dst[0], dst[1]))

visited = set()
visited.add((0, 0))

def getUnvisited():
    for src in sorted(fwd_map.keys()):
        if src not in visited:
            return src
    return None

p = getUnvisited()
assert p is not None

print("X = ROT(A%d%d, %2d);      // INSN: ROLI" % (p[0], p[1], rot_table[p[0]][p[1]]))
final_dst = fwd_map[p]
visited.add(p)

while True:
    dst = p
    p = bwd_map[p]
    if p in visited: break
    print("A%d%d = ROT(A%d%d, %2d);    // INSN: ROLI" % (dst[0], dst[1], p[0], p[1], rot_table[p[0]][p[1]]))
    visited.add(p)

print("A%d%d = X;               // INSN: MV" % (final_dst[0], final_dst[1]))

p = getUnvisited()
assert p is None
