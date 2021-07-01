import time

import matplotlib.pyplot as plt
import numpy as np
from ..define import Name, Operator
from ..utils import Utils


class Ga:
    def __init__(self, pop_size, max_generation, rc, rm, problem, func, max_or_min=0):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.rc = rc
        self.rm = rm
        self.problem = problem
        self.func = func
        self.max_or_min = max_or_min
        self.best = [None, None, None, []]  # (info, objective, fitness, tabu)
        self.pop = [[], [], []]  # (info, objective, fitness)
        # (start, end, best_objective, best_fitness, worst_fitness, mean_fitness)
        self.record = [[], [], [], [], [], []]
        self.max_tabu = Utils.len_tabu(self.problem.n)
        self.tabu = [[] for _ in range(self.pop_size)]

    def clear(self):
        self.best = [None, None, None, []]
        self.pop = [[], [], []]
        self.record = [[], [], [], [], [], []]
        self.tabu = [[] for _ in range(self.pop_size)]

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

    def append_individual(self, i, info):
        fit = Utils.calculate_fitness(self.max_or_min, info.obj)
        self.pop[0].append(info)
        self.pop[1].append(info.obj)
        self.pop[2].append(fit)
        self.tabu.append([])

    def replace_individual(self, i, info):
        fit = Utils.calculate_fitness(self.max_or_min, info.obj)
        if Utils.update(self.max_or_min, self.pop[1][i], info.obj):
            self.pop[0][i] = info
            self.pop[1][i] = info.obj
            self.pop[2][i] = fit
            self.tabu[i] = []

    def init_best(self):
        if self.max_or_min == 0:
            index = self.pop[1].index(max(self.pop[1]))
        else:
            index = self.pop[1].index(min(self.pop[1]))
        self.best[2] = self.pop[2][index]
        self.best[1] = self.pop[1][index]
        self.best[0] = self.pop[0][index]
        self.best[3] = self.tabu[index]

    def update_best(self):
        self.init_best()

    def show_generation(self, g):
        self.record[2].append(self.best[1])
        self.record[3].append(self.best[2])
        self.record[4].append(min(self.pop[2]))
        self.record[5].append(np.mean(self.pop[2]))
        Utils.print(
            "Generation {:<4} Runtime {:<8.4f} fBest: {:<.8f}, fWorst: {:<.8f}, fMean: {:<.8f}, gBest: {:<.2f} ".format(
                g, self.record[1][g] - self.record[0][g], self.record[3][g], self.record[4][g], self.record[5][g],
                self.record[2][g]))

    def selection_roulette(self):
        a = np.array(self.pop[2]) / sum(self.pop[2])
        b = np.array([])
        for i in range(a.shape[0]):
            b = np.append(b, sum(a[:i + 1]))
        pop = self.pop
        tabu = self.tabu
        self.pop = [[], [], []]
        self.tabu = [[] for _ in range(self.pop_size)]
        for i in range(self.pop_size):
            j = np.argwhere(b > np.random.random())[0, 0]
            self.pop[0].append(pop[0][j])
            self.pop[1].append(pop[1][j])
            self.pop[2].append(pop[2][j])
            self.tabu[i] = tabu[j]

    def selection_champion2(self):
        pop = self.pop
        tabu = self.tabu
        self.pop = [[], [], []]
        self.tabu = [[] for _ in range(self.pop_size)]
        for i in range(self.pop_size):
            a, b = np.random.choice(range(len(pop[0])), 2, replace=False)
            j = a if pop[2][a] > pop[2][b] else b
            self.pop[0].append(pop[0][j])
            self.pop[1].append(pop[1][j])
            self.pop[2].append(pop[2][j])
            self.tabu[i] = tabu[j]

    def do_selection(self):
        self.update_best()
        if self.problem.operator[Name.ga_s] in [Operator.default, Operator.ga_s_roulette]:
            self.selection_roulette()
        else:
            self.selection_champion2()
        self.pop[0][0] = self.best[0]
        self.pop[1][0] = self.best[1]
        self.pop[2][0] = self.best[2]
        self.tabu[0] = self.best[3]

    def do_init(self):
        pass

    def do_crossover(self, i, j):
        pass

    def do_mutation(self, i):
        pass

    def do_tabu(self, i):
        pass

    def decode(self, code):
        pass

    def do_dislocation(self, i, direction=0):
        code1 = self.pop[0][i].dislocation_operator(direction)
        self.append_individual(i, self.decode(code1))

    def do_evolution(self):
        Utils.print("{}Evolution  start{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTYELLOW_EX)
        self.clear()
        self.do_init()
        self.do_selection()
        for g in range(1, self.max_generation + 1):
            self.record[0].append(time.perf_counter())
            for i in range(self.pop_size):
                if self.problem.operator[Name.ts]:
                    self.do_tabu(i)
                if np.random.random() < self.rc:
                    j = np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)[0]
                    if self.problem.operator[Name.do] and Utils.similarity(self.pop[0][i].code,
                                                                           self.pop[0][j].code) >= 0.5:
                        self.do_dislocation(i, direction=0)
                        self.do_dislocation(j, direction=1)
                    else:
                        self.do_crossover(i, j)
                if np.random.random() < self.rm:
                    self.do_mutation(i)
            self.do_selection()
            self.record[1].append(time.perf_counter())
            self.show_generation(g)
        Utils.print("{}Evolution finish{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTRED_EX)


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
            info = self.decode(code)
            fit = Utils.calculate_fitness(self.max_or_min, info.obj)
            self.pop[0].append(info)
            self.pop[1].append(info.obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        code1, code2 = self.pop[0][i].ga_crossover(self.pop[0][j])
        self.append_individual(i, self.decode(code1))
        self.append_individual(j, self.decode(code2))

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation()
        self.append_individual(i, self.decode(code1))

    def do_tabu(self, i):
        code1 = self.pop[0][i].ts_permutation(self.tabu[i], self.max_tabu)
        self.replace_individual(i, self.decode(code1))
        if len(self.tabu[i]) >= self.max_tabu:
            self.tabu[i] = []
