import time

import numpy as np

from ..utils import Utils


class Sa:
    def __init__(self, pop_size, t0, t_min, alpha, problem, func, max_or_min=0):
        self.pop_size = pop_size
        self.t0 = t0
        self.t_min = t_min
        self.alpha = alpha
        self.t = t0
        self.problem = problem
        self.func = func
        self.max_or_min = max_or_min
        self.best = [None, None, None]  # (info, obj, fitness)
        self.pop = [[], [], []]  # (info, objective,fitness)
        self.record = [[], [], [], []]  # (start, end, objective, fitness)

    def clear(self):  # 开始进化前要进行清空, 方便快速地进行多次独立运行
        self.t = self.t0
        self.best = [None, None, None]
        self.pop = [[], [], []]
        self.record = [[], [], [], []]

    def update_t(self):
        self.t *= self.alpha

    def update_info(self, i, fit_new):
        p = np.exp(-np.abs((fit_new - self.pop[2][i])) / self.t)
        return True if np.random.random() < p else False

    def update_individual(self, i, info_new):  # 更新个体
        fit_new = Utils.calculate_fitness(self.max_or_min, info_new.obj)
        if Utils.update(self.max_or_min, self.pop[1][i], info_new.obj) or self.update_info(i, fit_new):
            self.pop[0][i] = info_new
            self.pop[1][i] = info_new.obj
            self.pop[2][i] = fit_new
        if Utils.update(self.max_or_min, self.best[1], info_new.obj):  # 更新最优个体
            self.best[0] = info_new
            self.best[1] = info_new.obj
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
        Utils.print("Generation: {:>4}, t: {:.4f}, runtime: {:.4f}, objective: {:>.4f}".format(
            g, self.t, self.record[1][g] - self.record[0][g], self.record[2][g]))

    def do_init(self):  # 定义初始化操作
        pass

    def decode_update(self, i, code):  # 解码并更新个体
        pass

    def do_update_individual(self, i):
        pass

    def do_evolution(self):  # 进化框架
        Utils.print("{}Evolution  start{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTYELLOW_EX)
        self.clear()
        self.do_init()
        g = 1
        while self.t > self.t_min:
            self.record[0].append(time.perf_counter())
            for i in range(self.pop_size):
                self.do_update_individual(i)
            self.record[1].append(time.perf_counter())
            self.update_t()
            self.show_generation(g)
            g += 1
        Utils.print("{}Evolution finish{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTRED_EX)


class SaNumericOptimization(Sa):
    def __init__(self, pop_size, t0, t_min, alpha, problem, func, max_or_min=0):
        Sa.__init__(self, pop_size, t0, t_min, alpha, problem, func, max_or_min)

    def decode(self, code):
        return self.problem.decode(self.func, code)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.problem.code(self.problem.low, self.problem.high, self.problem.dtype)
            else:
                code = pop[0][i].code
            info = self.decode(code)
            fit = Utils.calculate_fitness(self.max_or_min, info.obj)
            self.pop[0].append(info)
            self.pop[1].append(info.obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_update_individual(self, i):
        code = self.pop[0][i].sa_update(self.t)
        self.update_individual(i, self.decode(code))
