import copy
import time

import numpy as np

from .utils import Utils

deepcopy = copy.deepcopy


class De:
    def __init__(self, pop_size, max_generation, f, cr, problem, func, max_or_min=0):
        self.fl, self.fu = 0.1, 0.9
        self.crl, self.cru = 0.1, 0.6
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.f = f
        self.cr = cr
        self.problem = problem
        self.func = func
        self.max_or_min = max_or_min
        self.best = [None, None, None]  # (info, obj, fitness)
        self.pop = [[], [], []]  # (info, objective,fitness)
        self.record = [[], [], [], []]  # (start, end, objective, fitness)

    def clear(self):  # 开始进化前要进行清空, 方便快速地进行多次独立运行
        self.best = [None, None, None]
        self.pop = [[], [], []]
        self.record = [[], [], [], []]

    def update_individual(self, i, obj_new, info_new):  # 更新个体
        fit_new = Utils.calculate_fitness(self.max_or_min, obj_new)
        if Utils.update_accept_equal(self.max_or_min, self.pop[1][i], obj_new):
            self.pop[0][i] = info_new
            self.pop[1][i] = obj_new
            self.pop[2][i] = fit_new
        if Utils.update(self.max_or_min, self.best[1], obj_new):  # 更新最优个体
            self.best[0] = info_new
            self.best[1] = obj_new
            self.best[2] = fit_new

    def init_best(self):  # 初始化最优个体
        if self.max_or_min == 0:
            index = self.pop[1].index(max(self.pop[1]))
        else:
            index = self.pop[1].index(min(self.pop[1]))
        self.best[2] = self.pop[2][index]
        self.best[1] = self.pop[1][index]
        self.best[0] = self.pop[0][index]

    def show_generation(self, g):  # 显示进化信息
        self.record[2].append(self.best[1])
        self.record[3].append(self.best[2])
        Utils.print("Generation: {:>4}, runtime: {:.4f}, objective: {:>.4f}".format(
            g, self.record[1][g] - self.record[0][g], self.record[2][g]))

    def adaptive_f(self, i, j, k):  # 自适应f
        a = [i, j, k]
        b = [self.pop[2][v] for v in a]
        c = np.argsort(b)
        a, b = [a[v] for v in c], [b[v] for v in c]
        if b[0] != b[2]:  # b[0],b[1],b[2]:f_worst,f_middle,f_best
            self.f = self.fl + (self.fu - self.fl) * ((b[1] - b[2]) / (b[0] - b[2]))
        else:
            self.f = self.fu - self.fl
        return a

    def adaptive_cr(self, i):  # 自适应cr
        f = self.pop[2][i]
        f_max = max(self.pop[2])
        f_min = min(self.pop[2])
        if f_max != f_min:
            self.cr = self.crl + (self.cru - self.crl) * ((f - f_min) / (f_max - f_min))
        else:
            self.cr = self.cru - self.crl

    def do_init(self):  # 定义初始化操作
        pass

    def decode_update(self, i, code):  # 解码并更新个体
        pass

    def do_mutation(self, i):  # 变异操作
        pass

    def do_crossover(self, i, j):  # 交叉操作
        pass

    def do_evolution(self):  # 进化框架
        Utils.print("{}Evolution  start{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTYELLOW_EX)
        self.clear()
        self.do_init()
        for g in range(1, self.max_generation + 1):
            self.record[0].append(time.perf_counter())
            for i in range(self.pop_size):
                self.do_mutation(i)
                j = np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)[0]
                self.do_crossover(i, j)
            self.record[1].append(time.perf_counter())
            self.show_generation(g)
        Utils.print("{}Evolution finish{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTRED_EX)


class DeNumericOptimization(De):
    def __init__(self, pop_size, max_generation, f, cr, problem, func, max_or_min=0):
        De.__init__(self, pop_size, max_generation, f, cr, problem, func, max_or_min)

    def decode_update(self, i, code):
        info = self.problem.decode(self.func, code)
        self.update_individual(i, info.obj, info)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.problem.code(self.problem.low, self.problem.high, self.problem.dtype)
            else:
                code = pop[0][i].code
            info = self.problem.decode(self.func, code)
            fit = Utils.calculate_fitness(self.max_or_min, info.obj)
            self.pop[0].append(info)
            self.pop[1].append(info.obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_mutation(self, i):
        j, k, p, q = np.random.choice(np.delete(np.arange(self.pop_size), i), 4, replace=False)
        i, j, k = self.adaptive_f(i, j, k)
        code1 = self.pop[0][i].de_mutation(self.f, self.pop[0][j], self.pop[0][k], self.pop[0][p], self.pop[0][q],
                                           self.best[0])
        self.decode_update(i, code1)

    def do_crossover(self, i, j):
        self.adaptive_cr(i)
        code1, code2 = self.pop[0][i].de_crossover(self.cr, self.pop[0][j])
        self.decode_update(i, code1)
        self.decode_update(j, code2)
