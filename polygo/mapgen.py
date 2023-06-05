#!/usr/bin/env python3

import click
from click import echo

import svgwrite
from svgwrite import rgb

import numpy as np
from types import SimpleNamespace
from collections import defaultdict
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

def fixpoint(p):
    r = np.array([
        min(+10000, max(-10000, p[0])),
        min(+10000, max(-10000, p[1]))
    ])
    if (r != p).any():
        echo(f"{p} -> {r}")
    return r

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

        echo(f"{len(self.points)} points and {len(self.regions)} regions before elimination.")

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
            #echo(f"{count}/{self.N} merging verices {a} and {b}, norm {d}.")
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

        echo(f"{len(self.points)} points and {len(self.regions)} regions after {count} elimination steps.")
        return count

    def balance(self, weight_keep=1.0, weight_sides=1.0, weight_angles=1.0):
        x0 = list()

        for p in self.points:
            x0.append(p[0])
            x0.append(p[1])

        def f(x, verbose=False):
            if verbose:
                dbg = defaultdict(list)
            y = list()

            for a, b in zip(x, x0):
                y.append(weight_keep * (a-b) / max(self.W, self.H))
                if verbose: dbg[f"keep:"].append(y[-1])

            for idx, r in enumerate(self.regions):
                for i in range(1, len(r)-1):
                    edges = [
                        ((x[2*a]-x[2*b])**2 + (x[2*a+1]-x[2*b+1])**2)**0.5
                                for a, b in zip(r, r[i:] + r[:i])
                    ]
                    for p, q in zip(edges, edges[1:] + edges[:1]):
                        y.append(weight_sides * ((p - q) / (p + q))**2 / len(r)**2)
                        if verbose: dbg[f"region {idx:3d} {i}-steps:"].append(y[-1])

                directions = [
                        (v := np.array([x[2*a]-x[2*b], x[2*a+1]-x[2*b+1]])) / np.linalg.norm(v)
                                for a, b in zip(r, r[i:] + r[:i])
                ]
                for p, q in zip(directions, directions[1:] + directions[:1]):
                    y.append(weight_angles / (1.0 - p.dot(q)))
                    if verbose: dbg[f"region {idx:3d} angles:"].append(y[-1])

            if verbose:
                for key, data in sorted(dbg.items()):
                    echo(f"{len(data):5d}x {key:20s} mean={np.mean(data):6.04}, std={np.std(data):6.04}, min={np.min(data):6.04}, max={np.max(data):6.04}")
            return y

        echo("Solving least-squares fit..")
        x, cov_x, info, msg, ier = scipy.optimize.leastsq(f, x0, epsfcn=1.0, full_output=True)
        echo(f"{msg=} {ier=}")
        # f(x, True)

        min_x, max_x = min(x[0::2]), max(x[0::2])
        min_y, max_y = min(x[1::2]), max(x[1::2])
        #echo(f"{min_x=}, {max_x=}, {min_y=}, {max_y=}")

        for i in range(len(self.points)):
            self.points[i][0] = self.delta + (self.W-2*self.delta) * (x[2*i  ]-min_x) / (max_x-min_x)
            self.points[i][1] = self.delta + (self.H-2*self.delta) * (x[2*i+1]-min_y) / (max_y-min_y)

    def writesvg(self, filename, stones=False):
        echo(f"Writing SVG file '{filename}'..")

        dwg = svgwrite.Drawing(filename, size=(self.W, self.H), profile='tiny')
        dwg.add(dwg.rect((0, 0), (self.W, self.H), fill=bg))

        if stones:
            V = scipy.spatial.Voronoi(self.points)
            p = list(np.random.choice(len(self.points), [50]))
            players = list(zip([red, blue], [light_red, light_blue], [p[:25], p[25:]]))

            for color, lcolor, points in players:
                for p in points:
                    r = V.point_region[p]
                    if r < 0: continue
                    k = [q for q in V.regions[r] if q >= 0]
                    if len(k) >= 3:
                        dwg.add(dwg.polygon([V.vertices[q] for q in k], fill=lcolor))

        for r in self.regions:
            for a, b in zip(r, r[1:] + r[:1]):
                dwg.add(dwg.line(fixpoint(self.points[a]), fixpoint(self.points[b]), stroke=grey))

        for p in self.points:
            dwg.add(dwg.circle(fixpoint(p), 4, fill=black))

        if stones:
            for color, lcolor, points in players:
                for p in points:
                    dwg.add(dwg.circle(self.points[p], self.delta/2, fill=color))

        dwg.save()

@click.command("mapgen")
@click.option("--maxiter", default=4, help="maximum number of loop iterations")
@click.option("--seed", type=int, help="RNG Seed Value")
@click.argument("prefix", default="map")
def main(maxiter, seed, prefix):
    """
        Generate a PolyGO map and store it in a SVG file.
    """

    if seed is not None:
        np.random.seed(seed)

    mymap = Map()

    index = 0
    keep_running = True
    mymap.generate()

    while keep_running and index < maxiter:
        echo()
        echo(f"Main loop iteration {index}:")
        mymap.writesvg(f"{prefix}{2*index}.svg")
        keep_running = mymap.eliminate()
        mymap.writesvg(f"{prefix}{2*index+1}.svg")
        mymap.balance()
        index += 1

    echo()
    echo(f"Final version if the map:")
    mymap.writesvg(f"{prefix}{2*index}.svg")
    mymap.writesvg(f"{prefix}{2*index+1}.svg", True)
    echo("DONE.")

if __name__ == '__main__':
    main()

