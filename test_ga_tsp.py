import matplotlib.pyplot as plt

from src import *


def distance(val):
    return val


# low: 变量的下界; high: 变量的上界; dtype: 变量的数据类型; func: 目标函数
# max_or_min: 最大化或最小化(最大化: 值取0, 最小化: 值取1)
instance = "map"
n, city = Utils.string2data_tsp_cn(tsp_benchmark.instance[instance], dtype=float)
# instance = "example"
# instance = "eil51"
# n, city = Utils.string2data_tsp(tsp_benchmark.instance[instance], dtype=float)
# 参数
func = distance
max_or_min = 1
my_problem = Tsp()
for i in range(n):
    my_problem.add_city(city[i][1], city[i][2])
pop_size = 50
max_generation = 500
rc, rm = 0.85, 0.15
a = GaTsp(pop_size, max_generation, rc, rm, my_problem, func, max_or_min=max_or_min)
plt.figure()
a.do_evolution()
a.tsp_figure()
plt.savefig("tsp-%s" % instance)
Utils.make_dir("./res")
Utils.save_record_to_csv("./res/ObjTrace", a.record)
print(a.best[0].code + 1, a.best[1])
