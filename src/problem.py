import numpy as np

from .info import Info
from .resource import City
from .resource import Code
from .define import Name, Operator


class Common:
    def __init__(self, low, high, dtype):
        self.low = low
        self.high = high
        self.dtype = dtype
        a = [Name.de, Name.jaya, Name.pso, Name.sa, Name.ga_x, Name.ga_m, Name.ga_s]
        self.operator = {Name.ts: False, Name.do: False}
        for k in a:
            self.operator[k] = Operator.default
        try:
            self.n = len(low)
            self.var_range = np.array([i - j for i, j in zip(high, low)])
        except TypeError:
            pass


class NumericOptimization(Code, Common):
    def __init__(self, low, high, dtype):
        Common.__init__(self, low, high, dtype)

    def decode(self, func, code):
        return Info(self, code, func(code))

    def decode_pso(self, func, code):
        return Info(self, code, func(code[0]))


class Tsp(Code, Common):
    def __init__(self, low=None, high=None, dtype=None):
        Common.__init__(self, low, high, dtype)
        self.city = {}

    @property
    def n(self):
        return len(self.city)

    def code_heuristic(self, a=None):
        if a is None:
            a = np.random.randint(0, self.n, 1)[0]
        b = np.array([a, ], dtype=int)
        c = [val for val in range(self.n) if val not in b]
        while b.shape[0] < self.n:
            d = []
            for i in c:
                d.append(self.distance(a, i))
            e = d.index(min(d))
            b = np.append(b, c[e])
            a = c[e]
            c.remove(c[e])
        return b

    def add_city(self, x, y, name=None, index=None):
        if index is None:
            index = self.n
        self.city[index] = City(x, y, name)

    def distance(self, i, j):
        a = np.sqrt((self.city[i].x - self.city[j].x) ** 2 + (self.city[i].y - self.city[j].y) ** 2)
        return np.round(a, 0)

    def decode_ga(self, func, code):
        b = [(code[-1], code[0])]
        for i, j in zip(code[:-1], code[1:]):
            b.append((i, j))
        c = 0
        for i, j in b:  # 城市i和城市j的距离
            c += self.distance(i, j)
        return Info(self, code, func(c))

    def decode(self, func, code):
        info = self.decode_ga(func, np.argsort(code))
        info.code = code
        return info

    def decode_pso(self, func, code):
        info = self.decode_ga(func, np.argsort(code[0]))
        info.code = code
        return info
