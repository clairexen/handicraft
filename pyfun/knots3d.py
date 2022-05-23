#!/usr/bin/env python3

import PyQt5
import madcad as mc
from madcad import vec3, normalize, length, distance

verbose = False
scene = mc.Scene()
engineCallback = None

app = PyQt5.QtWidgets.QApplication([])
if mc.settings.display['system_theme']:
    mc.settings.use_qt_colors()

view = mc.rendering.View(scene)
view.setMinimumSize(PyQt5.QtCore.QSize(600, 400))
view.setWindowTitle("Knots 3D")
view.center(mc.fvec3(-0.5, 3, 5))
view.look(mc.fvec3(0, 0, 0))
view.adjust(mc.Box(mc.fvec3(-1.5, -1.5, -0.5), mc.fvec3(1.5, 1.5, 0.5)))
view.show()

timer = PyQt5.QtCore.QTimer()
timer.setInterval(100)
timer.timeout.connect(lambda: None if engineCallback is None else engineCallback())
timer.start()


#%% Setup

class Point:
    def __init__(self, pos):
        self.pos = pos
        self.isCtrl = False
        self.isFixed = False
        self.isCompr = True

class Rope:
    ctrlSphereMesh = mc.icosphere(vec3(0, 0, 0), 0.2)
    ctrlSphereMesh.option({"color": vec3(0.4, 0.4, 0.2)})

    fixedSphereMesh = mc.icosphere(vec3(0, 0, 0), 0.25)
    fixedSphereMesh.option({"color": vec3(0.8, 0.2, 0.2)})

    def __init__(self, points, meshOptions=None):
        self.path = list()

        for p in points:
            if len(self.path):
                q = self.path[-1].pos
                d = distance(q, p)
                n = int((d*10) // 3)
                for i in range(n):
                    q = q + (d/n)*normalize(p-q)
                    self.path.append(Point(q))

            self.path.append(Point(p))
            self.path[-1].isCtrl = True

        self.path[0].isFixed = True
        self.path[-1].isFixed = True

        self.objs = list()
        self.meshOptions = meshOptions
        self.sphereMesh = None

    def getLength(self):
        l = 0
        for i in range(len(self.path)-1):
            l += distance(self.path[i].pos, self.path[i+1].pos)
        return l

    def getLinks(self):
        return [distance(self.path[i].pos, self.path[i+1].pos) for i in range(len(self.path)-1)]

    def compressLinks(self):
        i = 0
        ret = 0
        newPath = list()

        while i+2 < len(self.path):
            p0, p1, p2 = self.path[i:i+3]

            compress = (p0.isCompr or p2.isCompr) and not p1.isFixed

            if compress and p1.isCtrl:
                compress = p0.isCtrl or p2.isCtrl

            if compress:
                compress = distance(p0.pos, p1.pos) < 0.17 and \
                           distance(p1.pos, p2.pos) < 0.17

            newPath.append(p0)
            i += 1 + compress
            ret += compress

        while i < len(self.path):
            newPath.append(self.path[i])
            i += 1

        if len(newPath) < 3:
            return 0

        self.path = newPath
        return ret

    def renderSpheres(self, ctrlOnly=False):
        if not ctrlOnly and self.sphereMesh is None:
            self.sphereMesh = mc.icosphere(vec3(0, 0, 0), 0.2)
            self.sphereMesh.option(self.meshOptions)

        self.objs.clear()
        for p in self.path:
            mesh = None if ctrlOnly else self.sphereMesh
            if p.isCtrl:
                mesh = self.ctrlSphereMesh
            if p.isFixed:
                mesh = self.fixedSphereMesh
            if mesh is not None:
                s = mc.Solid(content=mesh)
                s.itransform(p.pos)
                self.objs.append(s)

        return self.objs

    def renderTube(self, withCtrl=False):
        wire = mc.Interpolated([self.path[i].pos for i in range(len(self.path)) if self.path[i].isCtrl])

        tubeCrosssection = mc.Circle((self.path[0].pos, normalize(self.path[1].pos-self.path[0].pos)), 0.15)
        tube = mc.tube(tubeCrosssection, wire, True, True)
        tube.option(self.meshOptions)

        if withCtrl:
            self.renderSpheres(ctrlOnly=True)
            self.objs.append(tube)
        else:
            self.objs.clear()
            self.objs.append(tube)

        return self.objs

    # unused
    def getKinematic(self):
        solids = self.renderSpheres()
        fixed = [solids[0], solids[-1]]

        csts = list()
        for i in range(len(solids)-1):
            a, b = solids[i:i+2]
            d = self.path[i+1].pos - self.path[i].pos
            csts.append(mc.Ball(a, b, 0.5*d, -0.5*d))

        return csts, fixed, solids


# unused
def createRopesKinematic(ropes):
    kins = [r.getKinematic() for r in ropes]
    csts, fixed, solids = [sum(x, []) for x in zip(kins)]
    kin = mc.Kinematic(csts, fixed, solids)
    # scene.sync({"kin": kin})
    return kin


#      3  2  1  0 -1 -2 -3
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

def pointsColumn(levo, knotnum, x, Y, Z):
    mapz = lambda n: 0 if n==0 else ((2*sum([levo, n>0, (knotnum >> (abs(n)-1)) & 1]))&2)-1
    return list([vec3(0.5*x, 0.5*y, 0.3*mapz(z)) for y, z in zip(Y, Z)])

def rope(levo, knotnum):
    A, H, P = pointsColumn(levo, knotnum,  3, [-2, 0, 2], [0, 0, 0])
    E, L    = pointsColumn(levo, knotnum,  2, [-1, 1   ], [-1, +2])
    B, I, Q = pointsColumn(levo, knotnum,  1, [-2, 0, 2], [0, +3, 0])
    F, M    = pointsColumn(levo, knotnum,  0, [-1, 1   ], [+4, -5])
    C, J, R = pointsColumn(levo, knotnum, -1, [-2, 0, 2], [0, +6, 0])
    G, N    = pointsColumn(levo, knotnum, -2, [-1, 1   ], [-7, +8])
    D, K, S = pointsColumn(levo, knotnum, -3, [-2, 0, 2], [0, 0, 0])

    if levo:
        return Rope([A, E, I, M, R, N, K, G, C, F, I*vec3(1,1,-1), L, P], {"color": vec3(0.1, 0.4, 0.2)})
    else:
        return Rope([D, G, J, M, Q, L, H, E, B, F, J*vec3(1,1,-1), N, S], {"color": vec3(0.1, 0.2, 0.4)})


knotnum = 2
rope1 = rope(True, knotnum)    # Left (green) rope
rope2 = rope(False, knotnum)   # Right (blue) rope


#%% Engine

engineCallback = lambda: None



def tighten(ropes, nrounds=30):
    if verbose: print(f"Tighten {nrounds} rounds:")

    oldLengths = [r.getLength() for r in ropes]

    def applyForces(forces, maxStep=0.01, minStep=0.001):
        maxForce = 0.0

        for j, r in enumerate(ropes):
            for k in range(len(r.path)):
                f = forces[j][k]
                if r.path[k].isFixed:
                    # this fixes X coordinates directly in forces table
                    f.x = 0
                maxForce = max(maxForce, length(f))

        if maxForce < minStep:
            return False

        for j, r in enumerate(ropes):
            for k in range(len(r.path)):
                if maxForce > maxStep:
                    r.path[k].pos += maxStep * forces[j][k] / maxForce
                else:
                    r.path[k].pos += forces[j][k]

        return True

    for i in range(nrounds):
        # Step 1: Contract rope by chaining solids

        pull_forces = [[vec3(0, 0, 0) for p in r.path] for r in ropes]

        for j, r in enumerate(ropes):
            for k in range(len(r.path)-1):
                d = r.path[k+1].pos - r.path[k].pos
                pull_forces[j][k] += 0.5 * d
                pull_forces[j][k+1] -= 0.5 * d

        if not applyForces(pull_forces):
            break

        # Step 2: Correct for intersections

        if True:
            # Boring dumb iterative approach

            correction_cycles = 0

            while True:
                push_forces = [[vec3(0, 0, 0) for p in r.path] for r in ropes]

                # prevent self-intersection
                for j, r in enumerate(ropes):
                    for k in range(len(r.path)):
                        for l in range(k+1, len(r.path)):
                            d = r.path[l].pos - r.path[k].pos
                            x = 0.15 if l == k+1 else 0.30 if l == k+2 else 0.35
                            if length(d) < x:
                                push_forces[j][k] -= 0.5 * (x-length(d)) * normalize(d)
                                push_forces[j][l] += 0.5 * (x-length(d)) * normalize(d)

                # prevent intersecting other ropes
                for j, r in enumerate(ropes):
                    for g, s in enumerate(ropes):
                        if g <= j: continue
                        for k in range(len(r.path)):
                            for l in range(len(s.path)):
                                d = s.path[l].pos - r.path[k].pos
                                x = 0.35
                                if length(d) < x:
                                    push_forces[j][k] -= 0.5 * (x-length(d)) * normalize(d)
                                    push_forces[g][l] += 0.5 * (x-length(d)) * normalize(d)

                if not applyForces(push_forces, 0.005):
                    break

                correction_cycles += 1

            if verbose: print(f"  completed round {i} with {correction_cycles} correction cycles")

        else:
            # Solving a system of equations

            edges = list()

            # prevent self-intersection
            for j, r in enumerate(ropes):
                for k in range(len(r.path)):
                    for l in range(k+2, len(r.path)):
                        d = r.path[l].pos - r.path[k].pos
                        if length(d) < 0.3:
                            push_forces[j][k] -= normalize(d)
                            push_forces[j][l] += normalize(d)

            # prevent intersecting other ropes
            for j, r in enumerate(ropes):
                for g, s in enumerate(ropes):
                    if g <= j: continue
                    for k in range(len(r.path)):
                        for l in range(len(s.path)):
                            d = s.path[l].pos - r.path[k].pos
                            if length(d) < 0.3:
                                push_forces[j][k] -= normalize(d)
                                push_forces[g][l] += normalize(d)


    for j, r in enumerate(ropes):
        if verbose: print(f"  rope #{j} length: {oldLengths[j]:.5f} => {r.getLength():.5f} ({r.getLength()-oldLengths[j]:+.5f})")

    for j, r in enumerate(ropes):
        links = r.getLinks()
        if verbose: print(f"  rope #{j} links (num/min/avg/max): {len(links)} {min(links):.2f} {sum(links)/len(links):.2f} {max(links):.2f}")

        if min(links) < 0.17:
            r.compressLinks()
            links = r.getLinks()
            if verbose: print(f"  rope #{j} links after compression: {len(links)} {min(links):.2f} {sum(links)/len(links):.2f} {max(links):.2f}")


frameCount = 0

def theEngineCallback():
    global frameCount, knotnum, rope1, rope2

    if False:
        if frameCount == 0:
            print(f"Testing knot #{knotnum}")

        if frameCount == 30:
            knotnum += 1
            rope1 = rope(True, knotnum)
            rope2 = rope(False, knotnum)
            frameCount = 0
        else:
            frameCount += 1

    tighten([rope1, rope2])

    if False:
        scene.sync([])  # hotfix for madcad bug
        scene.sync({
            "rope_1": rope1.renderSpheres(),
            "rope_2": rope2.renderSpheres(),
        })
    elif False:
        scene.sync([])  # hotfix for madcad bug
        scene.sync({
            "rope_1": rope1.renderTube(True),
            "rope_2": rope2.renderTube(True),
        })
    else:
        scene.sync({
            "rope_1": rope1.renderTube(),
            "rope_2": rope2.renderTube(),
        })

    view.update()

engineCallback = theEngineCallback


#%% Qt Event Loop

app.exec()
