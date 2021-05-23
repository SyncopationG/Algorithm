import copy
import time

import matplotlib.pyplot as plt
import numpy as np

from .utils import Utils

deepcopy = copy.deepcopy


class Ga:
    def __init__(self, pop_size, max_generation, rc, rm, problem, func, max_or_min=0):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.rc = rc
        self.rm = rm
        self.problem = problem
        self.func = func
        self.max_or_min = max_or_min
        self.best = [None, None, None]  # (info, objective, fitness)
        self.pop = [[], [], []]  # (info, objective, fitness)
        # (start, end, best_objective, best_fitness, worst_fitness, mean_fitness)
        self.record = [[], [], [], [], [], []]

    def clear(self):
        self.best = [None, None, None]
        self.pop = [[], [], []]
        self.record = [[], [], [], [], [], []]

    def tsp_figure(self):
        try:
            sequence = np.argsort(self.best[0].code)
            data_x, data_y = [], []
            for i in sequence:
                data_x.append(self.problem.city[i].x)
                data_y.append(self.problem.city[i].y)
            data_x.append(self.problem.city[sequence[0]].x)
            data_y.append(self.problem.city[sequence[0]].y)
            plt.title(r"${Distance=%.4f}$" % self.best[1])
            plt.plot(data_x, data_y, "o-")
        except AttributeError:
            pass

    def append_individual(self, info):
        self.pop[0].append(info)
        self.pop[1].append(info.obj)
        self.pop[2].append(Utils.calculate_fitness(self.max_or_min, info.obj))

    def replace_individual(self, i, info):
        self.pop[0][i] = info
        self.pop[1][i] = info.obj
        self.pop[2][i] = Utils.calculate_fitness(self.max_or_min, info.obj)

    def init_best(self):
        if self.max_or_min == 0:
            index = self.pop[1].index(max(self.pop[1]))
        else:
            index = self.pop[1].index(min(self.pop[1]))
        self.best[2] = self.pop[2][index]
        self.best[1] = self.pop[1][index]
        self.best[0] = self.pop[0][index]

    def update_best(self, info, obj_new, fit_new):
        if Utils.update(self.max_or_min, self.best[1], obj_new):
            self.best[0] = info
            self.best[1] = obj_new
            self.best[2] = fit_new

    def show_generation(self, g):
        self.record[2].append(self.best[1])
        self.record[3].append(self.best[2])
        self.record[4].append(min(self.pop[2]))
        self.record[5].append(np.mean(self.pop[2]))
        Utils.print(
            "Generation {:<4} Runtime {:<8.4f} fBest: {:<.8f}, fWorst: {:<.8f}, fMean: {:<.8f}, gBest: {:<.2f} ".format(
                g, self.record[1][g] - self.record[0][g], self.record[3][g], self.record[4][g], self.record[5][g],
                self.record[2][g]))

    def do_selection(self):
        a = np.array(self.pop[2]) / sum(self.pop[2])
        b = np.array([])
        for i in range(a.shape[0]):
            b = np.append(b, sum(a[:i + 1]))
        pop = deepcopy(self.pop)
        self.pop = [[], [], []]
        for i in range(self.pop_size):
            j = np.argwhere(b > np.random.random())[0, 0]
            self.pop[0].append(pop[0][j])
            self.pop[1].append(pop[1][j])
            self.pop[2].append(pop[2][j])
        self.pop[0][0] = self.best[0]
        self.pop[1][0] = self.best[1]
        self.pop[2][0] = self.best[2]

    def do_init(self):
        pass

    def do_crossover(self, i, j):
        pass

    def do_mutation(self, i):
        pass

    def do_evolution(self):
        Utils.print("{}Evolution  start{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTYELLOW_EX)
        self.clear()
        self.do_init()
        for g in range(1, self.max_generation + 1):
            self.record[0].append(time.perf_counter())
            self.do_selection()
            for i in range(self.pop_size):
                if np.random.random() < self.rc:
                    j = np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)[0]
                    self.do_crossover(i, j)
                if np.random.random() < self.rm:
                    self.do_mutation(i)
            self.record[1].append(time.perf_counter())
            self.show_generation(g)
        Utils.print("{}Evolution finish{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTRED_EX)


class GaNumericOptimization(Ga):
    def __init__(self, pop_size, max_generation, rc, rm, problem, func, max_or_min=0):
        Ga.__init__(self, pop_size, max_generation, rc, rm, problem, func, max_or_min)


class GaTsp(Ga):
    def __init__(self, pop_size, max_generation, rc, rm, problem, func, max_or_min=0):
        Ga.__init__(self, pop_size, max_generation, rc, rm, problem, func, max_or_min)

    def decode(self, code):
        return self.problem.decode_ga(self.func, code)

    def do_init(self, pop=None):
        self.record[0].append(time.perf_counter())
        for i in range(self.pop_size):
            if pop is None:
                if i < self.problem.n:
                    code = self.problem.code_heuristic(a=i)
                else:
                    code = self.problem.code_permutation(self.problem.n)
            else:
                code = pop[0][i].code
            self.append_individual(self.decode(code))
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover_pmx(self.pop[0][j])
        self.append_individual(self.decode(code1))
        self.append_individual(self.decode(code2))

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_tpe()
        self.append_individual(self.decode(code1))
