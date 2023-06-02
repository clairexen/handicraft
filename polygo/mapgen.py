import svgwrite
from svgwrite import rgb

import numpy as np
import scipy.spatial

W = 1000
H = W * 2**0.5

points = np.random.uniform(0, 1, (400,2))
points[:,0] *= W
points[:,1] *= H

V = scipy.spatial.Voronoi(points)
vertices = V.vertices
regions = V.regions

bg = rgb(255,255,255)
grey= rgb(100, 100, 100)
black = rgb(0, 0, 0)
red = rgb(200, 50, 50)
blue = rgb(50, 50, 2000)

dwg = svgwrite.Drawing('map.svg', size=(W, H), profile='tiny')
dwg.add(dwg.rect((0, 0), (W, H), fill=bg))

if False:
    for p in points:
        dwg.add(dwg.circle(p, 3, fill=black))
    
    for v in vertices:
        dwg.add(dwg.circle(v, 2, fill=grey))
        
    for r in regions:
        rr = [idx for idx in r if idx >= 0]
        for a, b in zip(rr, rr[1:] + rr[:1]):
            dwg.add(dwg.line(V.vertices[a], V.vertices[b], stroke=grey))

regions = [r for r in regions
           if len(r) >= 3 and min(r) >= 0 and 
           min([vertices[i][0] for i in r]) > 5 and
           min([vertices[i][1] for i in r]) > 5 and
           max([vertices[i][0] for i in r]) < W-5 and
           max([vertices[i][1] for i in r]) < H-5
]

def eliminate(epsilon):
    count = 0
    while True:
        shortest_a, shortest_b = None, None
        shortest_distance = W+H
        
        for r in regions:
            if len(r) < 3: continue
            rr = [idx for idx in r if idx >= 0]
            for a, b in zip(rr, rr[1:] + rr[:1]):
                A,B = vertices[a], vertices[b]
                d = np.linalg.norm(A-B)
                if d < 0.1:
                    continue
                if shortest_distance > d:
                    shortest_a = a
                    shortest_b = b
                    shortest_distance = d

        if shortest_distance > epsilon: break
        a, b, d = shortest_a, shortest_b, shortest_distance
    
        count += 1
        print(f"{count} merging verices {a} and {b}, norm {d}.")

        for r in regions:
            if len(r) < 3: continue
            if a in r:
                if b in r:
                    r.remove(a)
                    if len(r) < 3: r[:] = []
                else:
                    r[r.index(a)] = b
        
        ab = (vertices[a] + vertices[b]) / 2
        vertices[a] = None
        vertices[b] = ab

eliminate(30)

if True:
    points = set()
    for r in regions:
        rr = [idx for idx in r if idx >= 0]
        for a, b in zip(rr, rr[1:] + rr[:1]):
            points.add(tuple(vertices[a]))
            dwg.add(dwg.line(vertices[a], vertices[b], stroke=grey))
    for p in points:
        dwg.add(dwg.circle(p, 4, fill=black))

dwg.save()