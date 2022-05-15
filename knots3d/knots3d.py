#!/usr/bin/env python3
from madcad import *

class Rope:
    def __init__(self, points, meshOptions=None):
        self.path = list()
        self.pointIdx = list()
        for p in points:
            if len(self.path):
                q = self.path[-1]
                d = distance(q, p)
                n = int((d*10) // 3)
                for i in range(n):
                    q = q + (d/n)*normalize(p-q)
                    self.path.append(q)
            self.pointIdx.append(len(self.path))
            self.path.append(p)

        self.meshOptions = meshOptions

    def getLength(self):
        l = 0
        for i in range(len(self.path)-1):
            l += distance(self.path[i], self.path[i+1])
        return l

    def createScene(self):
        self.wire = Interpolated([self.path[i] for i in self.pointIdx])

        self.tubeCrosssection = Circle((self.path[0], normalize(self.path[1]-self.path[0])), 0.2)
        self.tube = tube(self.tubeCrosssection, self.wire, True, True)
        self.tube.option(self.meshOptions)

        self.sphereMesh = icosphere(vec3(0, 0, 0), 0.2)
        self.sphereMesh.option(self.meshOptions)

        self.solids = list()
        for p in self.path:
            s = Solid(content=self.sphereMesh)
            s.itransform(p)
            self.solids.append(s)

        self.fixed = [self.solids[0], self.solids[-1]]

        self.csts = list()
        for i in range(len(self.solids)-1):
            a, b = self.solids[i:i+2]
            d = self.path[i+1] - self.path[i];
            self.csts.append(Ball(a, b, 0.5*d, -0.5*d))


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
    return list([vec3(0.5*x, 0.5*y, z * (0.3 if levo else -0.3)) for y, z in zip(Y, Z)])

def rope(levo, ctrls):
    A, H, P = pointsColumn(levo, -3, [-2, 0, 2], [0, 0, 0])
    E, L    = pointsColumn(levo, -2, [-1, 1   ], [-1, +1])
    B, I, Q = pointsColumn(levo, -1, [-2, 0, 2], [0, +1, 0])
    F, M    = pointsColumn(levo,  0, [-1, 1   ], [+1, -1])
    C, J, R = pointsColumn(levo,  1, [-2, 0, 2], [0, +1, 0])
    G, N    = pointsColumn(levo,  2, [-1, 1   ], [-1, +1])
    D, K, S = pointsColumn(levo,  3, [-2, 0, 2], [0, 0, 0])

    if levo:
        return Rope([A, E, I, M, R, N, K, G, C, F, I*vec3(1,1,-1), L, P], {"color": vec3(0.1, 0.4, 0.2)})
    else:
        return Rope([D, G, J, M, Q, L, H, E, B, F, J*vec3(1,1,-1), N, S], {"color": vec3(0.1, 0.2, 0.4)})

def tighten(ropes, nrounds=1000):
    damping = 0.1
    for i in range(nrounds):
        print(f"Round #{i} (damping={damping}):")
        forces = list()

        # chaining of solids along a rope
        for j, r in enumerate(ropes):
            print(f"  rope #{j} length: {r.getLength()}")
            f = list([vec3(0, 0, 0) for p in r.path])
            for k in range(len(r.path)-1):
                d = r.path[k+1] - r.path[k]
                f[k] += d
                f[k+1] -= d
            forces.append(f)

        # prevent self-intersection
        for j, r in enumerate(ropes):
            for k in range(len(r.path)):
                for l in range(k+2, len(r.path)):
                    d = r.path[l] - r.path[k]
                    if length(d) < 0.3:
                        #print(f"  SELF-COLL {j}:{k} {j}:{l}")
                        forces[j][k] -= normalize(d)
                        forces[j][l] += normalize(d)

        # prevent intersecting other ropes
        for j, r in enumerate(ropes):
            for g, s in enumerate(ropes):
                if g <= j: continue
                for k in range(len(r.path)):
                    for l in range(len(s.path)):
                        d = s.path[l] - r.path[k]
                        if length(d) < 0.3:
                            #print(f"  CROSS-COLL {j}:{k} {g}:{l}")
                            forces[j][k] -= normalize(d)
                            forces[g][l] += normalize(d)

        # apply changes
        for j, r in enumerate(ropes):
            for k in range(len(r.path)):
                if k == 0 or k == len(r.path)-1:
                    continue
                f = forces[j][k]
                if length(f) > 1: f = normalize(f)
                r.path[k] += damping * f

        damping *= 0.995

rope1 = rope(True, [])
rope2 = rope(False, [])

print(f"Number of objects: {len(rope1.path)+len(rope2.path)}")

tighten([rope1, rope2])

rope1.createScene();
rope2.createScene();

show([rope1.tube, rope2.tube])
#show([rope1.solids, rope2.solids])

if False:
    kin = Kinematic(
        rope1.csts + rope2.csts,
        fixed = rope1.fixed + rope2.fixed,
        solids = rope1.solids + rope2.solids)
    show([kin])

