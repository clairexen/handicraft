#!/usr/bin/env python3

import PyQt5
import madcad as mc
from madcad import fvec3, vec3, normalize, length, distance

scene = mc.Scene()
engineCallback = None

app = PyQt5.QtWidgets.QApplication([])
if mc.settings.display['system_theme']:
    mc.settings.use_qt_colors()

view = mc.rendering.View(scene)
view.setMinimumSize(PyQt5.QtCore.QSize(600, 400))
view.setWindowTitle("Knots 3D")
view.center(fvec3(-0.5, 3, 5))
view.look(fvec3(0, 0, 0))
view.adjust(mc.Box(fvec3(-1.5, -1.5, -0.5), fvec3(1.5, 1.5, 0.5)))
view.show()

timer = PyQt5.QtCore.QTimer()
timer.setInterval(100)
timer.timeout.connect(lambda: None if engineCallback is None else engineCallback())
timer.start()


#%% Setup

class Rope:
    def __init__(self, points, meshOptions=None):
        self.path = list()
        self.objs = list()
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
        self.sphereMesh = None

    def getLength(self):
        l = 0
        for i in range(len(self.path)-1):
            l += distance(self.path[i], self.path[i+1])
        return l

    def renderTube(self):
        wire = mc.Interpolated([self.path[i] for i in self.pointIdx])

        tubeCrosssection = mc.Circle((self.path[0], normalize(self.path[1]-self.path[0])), 0.2)
        tube = mc.tube(tubeCrosssection, wire, True, True)
        tube.option(self.meshOptions)

        self.objs = [ tube ]
        return self.objs

    def renderSpheres(self):
        if self.sphereMesh is None:
            sphereMesh = mc.icosphere(vec3(0, 0, 0), 0.2)
            sphereMesh.option(self.meshOptions)

        self.objs = []
        for p in self.path:
            s = mc.Solid(content=sphereMesh)
            s.itransform(p)
            self.objs.append(s)

        return self.objs

    # unused
    def getKinematic(self):
        solids = self.renderSpheres()
        fixed = [solids[0], solids[-1]]

        csts = list()
        for i in range(len(solids)-1):
            a, b = solids[i:i+2]
            d = self.path[i+1] - self.path[i];
            csts.append(mc.Ball(a, b, 0.5*d, -0.5*d))

        return csts, fixed, solids


# unused
def createRopesKinematic(ropes):
    kins = [r.getKinematic() for r in ropes]
    csts, fixed, solids = [sum(x, []) for x in zip(kins)]
    kin = mc.Kinematic(csts, fixed, solids)
    # scene.sync({"kin": kin})
    return kin


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

rope1 = rope(True, [])
rope2 = rope(False, [])


#%% Engine

engineCallback = lambda: None

damping = 0.1

def tighten(ropes, nrounds=10):
    global damping

    print(f"Tighten {nrounds} rounds (damping={damping:.7f}):")

    oldLengths = [r.getLength() for r in ropes]

    for i in range(nrounds):
        forces = list()

        # chaining of solids along a rope
        for j, r in enumerate(ropes):
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

    for j, r in enumerate(ropes):
        print(f"  rope #{j} length: {oldLengths[j]:.5f} => {r.getLength():.5f} ({r.getLength()-oldLengths[j]:+.5f})")


def theEngineCallback():
    tighten([rope1, rope2])

    scene.sync({
        #"rope_1": rope1.renderSpheres(),
        #"rope_2": rope2.renderSpheres(),
        "rope_1": rope1.renderTube(),
        "rope_2": rope2.renderTube(),
    })

    view.update()

engineCallback = theEngineCallback


#%% Qt Event Loop

app.exec()
