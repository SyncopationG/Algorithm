import matplotlib.pyplot as plt
import numpy as np
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
# Example Test
# code_a = np.array([1, 3, 2, 0, 4, 5], dtype=int)
# code_b = np.array([3, 1, 4, 0, 5, 2], dtype=int)
# code_c = my_problem.code_heuristic(a=4)
# info_a = my_problem.decode_ga(distance, code_a)
# info_b = my_problem.decode_ga(distance, code_b)
# info_c = my_problem.decode_ga(distance, code_c)
# res_a, res_b = info_a.ga_crossover_heuristic_tsp(info_b, c=4)
# print(code_a, "# parent 1")
# print(code_b, "# parent 2")
# print(code_c, "# heuristic code --> c4")
# print(res_a, "# child 1")
# print(res_b, "# child 2")
# # Ga parameter
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
