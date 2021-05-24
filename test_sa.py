import numpy as np

from src import *


def booth_function(code):
    """
low = [-10, -10]
high = [10, 10]
dtype = [int, int]
    """
    x1, x2 = code
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def matyas_function(code):
    """
low = [-10, -10]
high = [10, 10]
dtype = [int, int]
        """
    x1, x2 = code
    return 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2


def mccormick_function(code):
    """
low = [-1.5, -3]
high = [4, 4]
dtype = [float, float]
        """
    x1, x2 = code
    return np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1


def six_hump_camel_function(code):
    """
low = [-3, -2]
high = [3, 2]
dtype = [float, float]
    """
    x1, x2 = code
    return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2


def fun_a(code):
    """
low = [-10, ]
high = [10, ]
dtype = [float, ]
    """
    x, = code
    return x + 10 * np.sin(5 * x) + 7 * np.cos(4 * x)


# low: 变量的下界; high: 变量的上界; dtype: 变量的数据类型; func: 目标函数
# max_or_min: 最大化或最小化(最大化: 值取0, 最小化: 值取1)
func = fun_a
low = [-10, ]
high = [10, ]
dtype = [float, ]
max_or_min = 0
my_problem = NumericOptimization(low, high, dtype)
# 参数
pop_size = 20
t0, t_min, alpha = 1000, 0.001, 0.95
my_problem.operator[Name.sa] = Operator.sa_classic
a = SaNumericOptimization(pop_size, t0, t_min, alpha, my_problem, func, max_or_min=max_or_min)
a.do_evolution()
a.best[0].print()
a.best[0].save("./Result/CodeObj")
Utils.save_record_to_csv("./Result/ObjTrace", a.record)
