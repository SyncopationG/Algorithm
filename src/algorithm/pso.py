import time

import numpy as np

from ..utils import Utils


class Pso:
    def __init__(self, pop_size, max_generation, c1, c2, w, problem, func, max_or_min=0):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.problem = problem
        self.func = func
        self.max_or_min = max_or_min
        self.pop = [[], []]  # (info, obj)
        self.pop_best = [[], []]  # (info, obj)
        self.best = [None, None]  # (info, obj)
        self.record = [[], [], [], []]  # (time_start, time_end, mean_obj, best_obj)

    def clear(self):
        self.pop = [[], []]  # (info, obj)
        self.best = [None, None]  # (info, obj)
        self.pop_best = [[], []]  # (info, obj)
        self.record = [[], [], [], []]  # (time_start, time_end, mean_obj, best_obj)

    def init_best(self):
        if self.max_or_min == 0:
            self.best[1] = max(self.pop[1])
        else:
            self.best[1] = min(self.pop[1])
        index = self.pop[1].index(self.best[1])
        self.best[0] = self.pop[0][index]

    def update(self, i, info):
        if Utils.update(self.max_or_min, self.pop_best[1][i], info.obj):
            self.pop_best[0][i] = info
            self.pop_best[1][i] = info.obj
        if Utils.update(self.max_or_min, self.best[1], info.obj):
            self.best[0] = info
            self.best[1] = info.obj
        self.pop[0][i], self.pop[1][i] = info, info.obj

    def show_generation(self, g):
        self.record[2].append(np.mean(self.pop_best[1]))
        self.record[3].append(self.best[1])
        Utils.print("Generation {:<4} Runtime {:<8.4f} meanObj: {:<.4f},  bestObj: {:<.4f} ".format(
            g, self.record[1][g] - self.record[0][g], self.record[2][g], self.record[3][g]))

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


class PsoNumericOptimization(Pso):
    def __init__(self, pop_size, max_generation, c1, c2, w, problem, func, max_or_min=0):
        Pso.__init__(self, pop_size, max_generation, c1, c2, w, problem, func, max_or_min)

    def decode(self, code):
        return self.problem.decode_pso(self.func, code)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                code = self.problem.code_particle(self.problem.low, self.problem.high, self.problem.dtype)
            else:
                code = pop[0][i].code
            info = self.problem.decode_pso(self.func, code)
            self.pop[0].append(info)
            self.pop[1].append(info.obj)
            self.pop_best[0].append(info)
            self.pop_best[1].append(info.obj)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_update_individual(self, i):
        code = self.pop[0][i].pso_update(self.c1, self.c2, self.w, self.pop_best[0][i], self.best[0])
        self.update(i, self.decode(code))
