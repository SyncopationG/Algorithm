import matplotlib.pyplot as plt

from src import *


def distance(val):
    return val


# low: 变量的下界; high: 变量的上界; dtype: 变量的数据类型; func: 目标函数
# max_or_min: 最大化或最小化(最大化: 值取0, 最小化: 值取1)
instance = "map"
n, city = Utils.string2data_tsp_cn(tsp_benchmark.instance[instance], dtype=float)
# instance = "eil51"
# n, city = Utils.string2data_tsp(tsp_benchmark.instance[instance], dtype=float)
print(n, city)
func = distance
low = [0] * n
high = [1] * n
dtype = [float] * n
max_or_min = 1
my_problem = Tsp(low, high, dtype)
for i in range(n):
    my_problem.add_city(city[i][1], city[i][2])
# De parameter
pop_size = 50
max_generation = 500
f, cr = 0.5, 0.5
a = DeNumericOptimization(pop_size, max_generation, f, cr, my_problem, func, max_or_min=max_or_min)
a.do_evolution()
Utils.make_dir("./res")
Utils.save_record_to_csv("./res/ObjTrace", a.record)
print(a.best[0].code, a.best[1])
