import svgwrite
from svgwrite import rgb

import numpy as np
from types import SimpleNamespace
import scipy.spatial
import scipy.optimize
import scipy.special

bg = rgb(255,255,255)
grey= rgb(100, 100, 100)
black = rgb(0, 0, 0)
red = rgb(200, 50, 50)
green = rgb(50, 200, 50)
blue = rgb(50, 50, 200)
light_red = rgb(250, 200, 200)
light_green = rgb(200, 250, 200)
light_blue = rgb(200, 200, 250)

class Map:
    def __init__(self, W=1000, H=1000*(2**0.5), N=200, delta=50):
        self.W, self.H, self.N = W, H, N
        self.delta = delta

    def generate(self):
        self.seeds = np.random.uniform(0, 1, (self.N,2))
        self.seeds[:,0] *= self.W-2*self.delta
        self.seeds[:,1] *= self.H-2*self.delta
        self.seeds[:,0] += self.delta
        self.seeds[:,1] += self.delta

        V = scipy.spatial.Voronoi(self.seeds)
        self.points = list(V.vertices)
        self.regions = list(V.regions)

    def eliminate(self):
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

        def remove_point_or_edge(a=None, b=None):
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
            remove_point_or_edge(a, b)

        used_points = set()
        for r in self.regions:
            used_points |= set(r)

        s, pi = 0, list()
        for i in range(len(self.points)):
            if i not in used_points:
                self.points[i] = None
            if self.points[i] is None:
                pi.append(None)
            else:
                pi.append(s)
                s += 1

        self.points = [p for p in self.points if p is not None]
        self.regions = [[pi[i] for i in r] for r in self.regions]

        print(f"{len(self.points)} points and {len(self.regions)} regions after elimination.")

    def balance(self):
        x0 = list()

        for p in self.points:
            x0.append(p[0])
            x0.append(p[1])

        def f(x):
            y = list()
            for a, b in zip(x, x0):
                y.append((a-b)/10000)
            for r in self.regions:
                for i in range(1, len(r)-1):
                    edges = [
                        ((x[2*a]-x[2*b])**2 + (x[2*a+1]-x[2*b+1])**2)**0.5
                                for a, b in zip(r, r[i:] + r[:i])
                    ]
                    for p, q in zip(edges, edges[1:] + edges[:1]):
                        y.append(((p - q) / (p + q))**2 / len(r)**2)
            return y

        print("Solving least-squares fit..")
        x, cov_x, info, msg, ier = scipy.optimize.leastsq(f, x0, epsfcn=1.0, full_output=True)
        print(f"{msg=} {ier=}")

        min_x, max_x = min(x[0::2]), max(x[0::2])
        min_y, max_y = min(x[1::2]), max(x[1::2])
        #print(f"{min_x=}, {max_x=}, {min_y=}, {max_y=}")

        for i in range(len(self.points)):
            self.points[i][0] = self.delta + (self.W-2*self.delta) * (x[2*i  ]-min_x) / (max_x-min_x)
            self.points[i][1] = self.delta + (self.H-2*self.delta) * (x[2*i+1]-min_y) / (max_y-min_y)

    def writesvg(self, filename, stones=False):
        print(f"Writing SVG file '{filename}'..")

        dwg = svgwrite.Drawing(filename, size=(self.W, self.H), profile='tiny')
        dwg.add(dwg.rect((0, 0), (self.W, self.H), fill=bg))

        if stones:
            V = scipy.spatial.Voronoi(self.points)
            p = list(np.random.choice(len(self.points), [30]))
            players = list(zip(
                [red, blue, green],
                [light_red, light_blue, light_green],
                [p[0:10], p[10:20], p[20:30]]
            ))

            for color, lcolor, points in players:
                for p in points:
                    r = V.point_region[p]
                    if r < 0: continue
                    k = [q for q in V.regions[r] if q >= 0]
                    if len(k) >= 3:
                        dwg.add(dwg.polygon([V.vertices[q] for q in k], fill=lcolor))

        for r in self.regions:
            for a, b in zip(r, r[1:] + r[:1]):
                dwg.add(dwg.line(self.points[a], self.points[b], stroke=grey))

        for p in self.points:
            dwg.add(dwg.circle(p, 4, fill=black))

        if stones:
            for color, lcolor, points in players:
                for p in points:
                    dwg.add(dwg.circle(self.points[p], self.delta/2, fill=color))

        dwg.save()

mymap = Map()

mymap.generate()
mymap.eliminate()
mymap.writesvg("map1.svg")

mymap.balance()
mymap.writesvg("map2.svg")

for i in range(3,6):
    mymap.eliminate()
    mymap.balance()
    mymap.writesvg(f"map{i}.svg")

mymap.writesvg("map6.svg", True)
print("DONE.")
