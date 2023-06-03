import svgwrite
from svgwrite import rgb

import numpy as np
from types import SimpleNamespace
import scipy.spatial

bg = rgb(255,255,255)
grey= rgb(100, 100, 100)
black = rgb(0, 0, 0)
red = rgb(200, 50, 50)
blue = rgb(50, 50, 2000)

class Map:
    def __init__(self, W=1000, H=1000*(2**0.5), N=200, delta=50):
        self.W, self.H, self.N = W, H, N
        self.delta = delta

    def generate(self):
        self.seeds = np.random.uniform(0, 1, (self.N,2))
        self.seeds[:,0] *= self.W
        self.seeds[:,1] *= self.H

        V = scipy.spatial.Voronoi(self.seeds)
        self.points = list(V.vertices)
        self.regions = list(V.regions)

        def check_point(p):
            if p[0] < 5 or p[0] > self.W-5: return False
            if p[1] < 5 or p[1] > self.H-5: return False
            return True

        def cleanup_region(index):
            r = self.regions[index]
            r = [i for i in r if i >= 0 and check_point(self.points[i])]
            if len(r) < 3:
                del self.regions[index]
                return None
            self.regions[index] = r
            return r

        def cleanup(a=None, b=None):
            for i in range(len(self.regions)-1, -1, -1):
                if a is not None and a in (r := self.regions[i]):
                    if b is None or b in r:
                        r.remove(a)
                    else:
                        r[r.index(a)] = b
                cleanup_region(i)
            if a is not None and b is not None:
                self.points[a], self.points[b] = None, (self.points[a] + self.points[b]) / 2

        print(f"{len(self.points)} points and {len(self.regions)} regions before elimination.")

        count = 0
        while True:
            shortest = SimpleNamespace(a=None, b=None, d=self.W+self.H)

            for r in self.regions:
                for a, b in zip(r, r[1:] + r[:1]):
                    d = np.linalg.norm(self.points[a] - self.points[b])
                    if shortest.d > d:
                        shortest.a = a
                        shortest.b = b
                        shortest.d = d

            if shortest.d > self.delta: break
            a, b, d = shortest.a, shortest.b, shortest.d
        
            count += 1
            #print(f"{count}/{self.N} merging verices {a} and {b}, norm {d}.")
            cleanup(a, b)

        s, pi = 0, list()
        for p in self.points:
            pi.append(s)
            if p is not None: s += 1

        self.points = [p for p in self.points if p is not None]
        self.regions = [[pi[i] for i in r] for r in self.regions]

        print(f"{len(self.points)} points and {len(self.regions)} regions after elimination.")

    def writesvg(self, filename):
        dwg = svgwrite.Drawing(filename, size=(self.W, self.H), profile='tiny')
        dwg.add(dwg.rect((0, 0), (self.W, self.H), fill=bg))

        pp = set()
        for r in self.regions:
            for a, b in zip(r, r[1:] + r[:1]):
                pp.add(tuple(self.points[a]))
                dwg.add(dwg.line(self.points[a], self.points[b], stroke=grey))
        for p in pp:
            dwg.add(dwg.circle(p, 4, fill=black))

        dwg.save()

mymap = Map()
mymap.generate()
mymap.writesvg("map.svg")

