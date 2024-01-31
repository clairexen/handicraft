
import Tkinter
import random
from math import *
from pga import *

init_node_order = [ ]
best_node_order = [ ]
best_node_order_badness = -1.0

genlen = 125
length = 121
points = [(0,0)] * length
for x in range(0,11):
    for y in range(0,11):
        i = x*11 + y
        points[i] = (60 * x - 310 + 20*random.random(), 60 * y - 310 + 20*random.random())

def get_node_order(pg, p, pop):
        node_order = [0] * length
        for i in range(0, length):
            node_order[i] = init_node_order[i]
        for i in range(0, genlen):
            a = pg.get_allele(p, pop, i*2+0)
            b = pg.get_allele(p, pop, i*2+1)
            tmp = node_order[a]
            node_order[a] = node_order[b]
            node_order[b] = tmp
        return node_order

def optimize():
    class MyPGA(PGA):
        def evaluate(self, p, pop):
            global best_node_order_badness
            global best_node_order
            bad = 0.0
            node_order = get_node_order(self, p, pop)
            pos = points[node_order[0]]
            for i in range(1, length):
                bad = bad + sqrt((pos[0]-points[node_order[i]][0])**2 + (pos[1]-points[node_order[i]][1])**2)
                pos = points[node_order[i]]
            if best_node_order_badness < 0 or best_node_order_badness > bad:
                best_node_order_badness = bad
                best_node_order = node_order
            return bad

    pg = MyPGA(int, genlen*2, 
            init = [(0,length-1)] * (genlen*2),
            maximize = False
    )
    pg.run()

    for i in range(0, length):
        init_node_order[i] = best_node_order[i]

init_node_order = [0] * length
for i in range(0, length):
    init_node_order[i] = i

optimize()
print "=== All time best badness: {0} ===".format(best_node_order_badness)

tk = Tkinter.Tk()
tk.wm_geometry("700x700")

canvas = Tkinter.Canvas(tk, width=700, height=700)
canvas.pack()

for i in range(1, length):
    canvas.create_line(350+points[best_node_order[i-1]][0], 350+points[best_node_order[i-1]][1],
            350+points[best_node_order[i]][0], 350+points[best_node_order[i]][1])

for i in range(0, length):
    canvas.create_oval(345 + points[i][0], 345 + points[i][1],
            355 + points[i][0], 355 + points[i][1], outline="green")

tk.mainloop()

