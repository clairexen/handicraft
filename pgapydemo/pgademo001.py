
from pga import *

length = 10

class MyPGA(PGA):
    def evaluate(self, p, pop):
        bad = 0.0
        for i in range(1, length):
            a = self.get_allele(p, pop, i-1)
            b = self.get_allele(p, pop, i)
            bad = bad + abs(b - a - 1)
        return bad

pg = MyPGA(float, length, 
        init = [(0.0,20.0)] * length,
        maximize = False
)
pg.run()

