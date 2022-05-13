#!/usr/bin/env python3
from madcad import show, vec3
from madcad.mathutils import distance, normalize
from madcad.generation import icosphere


class Rope:
    def __init__(self, points):
        self.path = list()
        for p in points:
            if len(self.path):
                q = self.path[-1]
                d = distance(q, p)
                n = int((d*10) // 3)
                for i in range(n):
                    q = q + (d/n)*normalize(p-q)
                    self.path.append(q)
            self.path.append(p)   


#     -3 -2 -1  0  1  2  3
#
#  -2  A     B     C     D
#       \   / \   / \   /
#        \ /   \ /   \ /
#  -1     E     F     G
#        / \   / \   / \
#       /   \ /   \ /   \
#   0  H     I     J     K
#       \   / \   / \   /
#        \ /   \ /   \ /
#   1     L     M     N
#        / \   / \   / \
#       /   \ /   \ /   \
#   2  P     Q     R     S

def pointsColumn(levo, x, Y, Z):
    return list([vec3(x, y, z * (0.5 if levo else -0.5)) for y, z in zip(Y, Z)])

def rope(levo, ctrls):
    A, H, P = pointsColumn(levo, -3, [-2, 0, 2], [0, 0, 0])
    E, L    = pointsColumn(levo, -2, [-1, 1   ], [-1, +1])
    B, I, Q = pointsColumn(levo, -1, [-2, 0, 2], [0, +1, 0])
    F, M    = pointsColumn(levo,  0, [-1, 1   ], [+1, -1])
    C, J, R = pointsColumn(levo,  1, [-2, 0, 2], [0, +1, 0])
    G, N    = pointsColumn(levo,  2, [-1, 1   ], [-1, +1])
    D, K, S = pointsColumn(levo,  3, [-2, 0, 2], [0, 0, 0])

    if levo:
        return Rope([A, E, I, M, R, N, K, G, C, F, I*vec3(1,1,-1), L, P])
    else:
        return Rope([D, G, J, M, Q, L, H, E, B, F, J*vec3(1,1,-1), N, S])


rope1 = rope(True, [])
rope2 = rope(False, [])

meshes = list()
for p in rope1.path:
    s = icosphere(p, 0.2)
    s.option(color=vec3(0.0,0.2,0.4))
    meshes.append(s)

for p in rope2.path:
    s = icosphere(p, 0.2)
    s.option(color=vec3(0.0,0.4,0.2))
    meshes.append(s)

show(meshes)
