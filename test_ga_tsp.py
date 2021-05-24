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
max_or_min = 1
my_problem = Tsp()
for i in range(n):
    my_problem.add_city(city[i][1], city[i][2])
# 参数
func = distance
pop_size = 20
max_generation = 100
rc, rm = 0.85, 0.15
my_problem.operator[Name.ga_x] = Operator.ga_x_tsp_h
my_problem.operator[Name.ga_m] = Operator.ga_m_tpe
my_problem.operator[Name.ga_s] = Operator.ga_s_roulette
my_problem.operator[Name.ts] = True
my_problem.operator[Name.do] = True
a = GaTsp(pop_size, max_generation, rc, rm, my_problem, func, max_or_min=max_or_min)
a.do_evolution()
plt.figure()
a.tsp_figure()
plt.savefig("./Result/tsp-%s" % instance)
a.best[0].print()
a.best[0].save("./Result/CodeObj")
Utils.save_record_to_csv("./Result/ObjTrace", a.record)
