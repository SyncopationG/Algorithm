import time

import numpy as np

from ..utils import Utils


class Jaya:
    def __init__(self, pop_size, max_generation, problem, func, max_or_min=0):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.problem = problem
        self.func = func
        self.max_or_min = max_or_min
        self.pop = [[], []]  # (info, obj)
        self.best = [None, None]  # (info, obj)
        self.worst = [None, None]  # (info, obj)
        self.record = [[], [], [], [], []]  # (time_start, time_end, worst, mean_obj, best_obj)

    def clear(self):
        self.pop = [[], []]  # (info, obj)
        self.best = [None, None]  # (info, obj)
        self.worst = [None, None]  # (info, obj)
        self.record = [[], [], [], [], []]  # (time_start, time_end, mean_obj, worst_obj, best_obj)

    def init_best(self):
        if self.max_or_min == 0:
            self.best[1] = max(self.pop[1])
        else:
            self.best[1] = min(self.pop[1])
        index = self.pop[1].index(self.best[1])
        self.best[0] = self.pop[0][index]

    def update_best(self, info):
        if Utils.update(self.max_or_min, self.best[1], info.obj):
            self.best[0] = info
            self.best[1] = info.obj

    def update_worst(self):
        if self.max_or_min == 0:
            self.worst[1] = min(self.pop[1])
        else:
            self.worst[1] = max(self.pop[1])
        index = self.pop[1].index(self.worst[1])
        self.worst[0] = self.pop[0][index]

    def update(self, i, info):
        if Utils.update(self.max_or_min, self.pop[1][i], info.obj):
            self.pop[0][i] = info
            self.pop[1][i] = info.obj
        self.update_best(info)

    def show_generation(self, g):
        self.update_worst()
        self.record[2].append(np.mean(self.pop[1]))
        self.record[3].append(self.worst[1])
        self.record[4].append(self.best[1])
        Utils.print("Generation {:<4} Runtime {:<8.4f} meanObj: {:<.4f}, worstObj: {:<.4f}, bestObj: {:<.4f} ".format(
            g, self.record[1][g] - self.record[0][g], self.record[2][g], self.record[3][g], self.record[4][g]))

    def do_init(self, pop=None):
        pass

    def do_update_individual(self, i):
        pass

    def do_evolution(self, pop=None):
        Utils.print("{}Evolution  start{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTYELLOW_EX)
        self.clear()
        self.do_init(pop)
        for g in range(1, self.max_generation + 1):
            self.record[0].append(time.perf_counter())
            for i in range(self.pop_size):
                self.do_update_individual(i)
            self.record[1].append(time.perf_counter())
            self.show_generation(g)
        Utils.print("{}Evolution finish{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTRED_EX)


class JayaNumericOptimization(Jaya):
    def __init__(self, pop_size, max_generation, problem, func, max_or_min=0):
        Jaya.__init__(self, pop_size, max_generation, problem, func, max_or_min)

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
            self.pop[0].append(info)
            self.pop[1].append(info.obj)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_update_individual(self, i):
        j = np.random.choice(np.delete(range(self.pop_size), i), 1, replace=False)[0]
        code = self.pop[0][i].jaya_update(self.best[0], self.worst[0], self.pop[0][j])
        self.update(i, self.decode(code))
