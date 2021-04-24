import copy
import time

import matplotlib.pyplot as plt
import numpy as np

from .utils import Utils

deepcopy = copy.deepcopy
k1, k2, k3, k4 = 1, 1, 0.5, 0.5


class Ga:
    def __init__(self, pop_size, max_generation, rc, rm, problem, func, max_or_min=0):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.rc = rc
        self.rm = rm
        self.t0 = 1000
        self.t = 1000
        self.alpha = 0.95
        self.problem = problem
        self.func = func
        self.max_or_min = max_or_min
        self.best = [None, None, None]  # (info, objective, fitness)
        self.pop = [[], [], []]  # (info, objective, fitness)
        # (start, end, best_objective, best_fitness, worst_fitness, mean_fitness)
        self.record = [[], [], [], [], [], []]

    def clear(self):
        self.t = self.t0
        self.best = [None, None, None]
        self.pop = [[], [], []]
        self.record = [[], [], [], [], [], []]

    def dislocation(self, i):
        pass

    def update_t(self):
        self.t *= self.alpha

    def update_info(self, i, fit_new):
        p = np.exp(-np.abs(fit_new - self.pop[2][i]) / self.t)
        return True if np.random.random() < p else False

    def tsp_figure(self):
        try:
            # pass
            # """
            # plt.clf()
            sequence = np.argsort(self.best[0].code)
            data_x, data_y = [], []
            for i in sequence:
                data_x.append(self.problem.city[i].x)
                data_y.append(self.problem.city[i].y)
            data_x.append(self.problem.city[sequence[0]].x)
            data_y.append(self.problem.city[sequence[0]].y)
            plt.title(r"${Distance=%.4f}$" % self.best[1])
            plt.plot(data_x, data_y, "o-")
            # plt.pause(0.01)
            # """
        except AttributeError:
            pass

    def update_individual(self, i, obj_new, info_new):
        fit_new = Utils.calculate_fitness(self.max_or_min, obj_new)
        # self.pop[0].append(info_new)
        # self.pop[1].append(obj_new)
        # self.pop[2].append(fit_new)
        if Utils.update(self.max_or_min, self.pop[1][i], obj_new) or np.random.random() < 0.005:
            # if Utils.update(self.max_or_min, self.pop[1][i], obj_new) or self.update_info(i, fit_new):
            # if Utils.update(self.max_or_min, self.pop[1][i], obj_new):
            self.pop[0][i] = info_new
            self.pop[1][i] = obj_new
            self.pop[2][i] = fit_new
        if self.pop[0][i].similarity(info_new) >= 0.5:
            self.dislocation(i)
        if Utils.update(self.max_or_min, self.best[1], obj_new):
            self.best[0] = info_new
            self.best[1] = obj_new
            self.best[2] = fit_new
            # self.tsp_figure()

    def init_best(self):
        if self.max_or_min == 0:
            index = self.pop[1].index(max(self.pop[1]))
        else:
            index = self.pop[1].index(min(self.pop[1]))
        self.best[2] = self.pop[2][index]
        self.best[1] = self.pop[1][index]
        self.best[0] = self.pop[0][index]
        # self.tsp_figure()

    def adaptive_rc_rm_s(self, i, j):
        f_max, f_avg = max(self.pop[2]), np.mean(self.pop[2])
        f = max([self.pop[2][i], self.pop[2][j]])
        rc, rm = k2, k4
        if f > f_avg:
            rc = k1 * np.sin(np.pi * (f_max - f) / (2 * (f_max - f_avg)))
        if self.pop[2][i] > f_avg:
            rm = k3 * np.sin(np.pi * (f_max - self.pop[2][i]) / (f_max - f_avg))
        return rc, rm

    def adaptive_rc_rm_c(self, i, j):
        f_max, f_avg = max(self.pop[2]), np.mean(self.pop[2])
        f = max([self.pop[2][i], self.pop[2][j]])
        rc, rm = k2, k4
        if f > f_avg:
            rc = 1 - k1 * np.cos(np.pi * (f_max - f) / (2 * (f_max - f_avg)))
        if self.pop[2][i] > f_avg:
            rm = 1 - k3 * np.cos(np.pi * (f_max - self.pop[2][i]) / (f_max - f_avg))
        return rc, rm

    def show_generation(self, g):
        self.record[2].append(self.best[1])
        self.record[3].append(self.best[2])
        self.record[4].append(min(self.pop[2]))
        self.record[5].append(np.mean(self.pop[2]))
        Utils.print(
            "Generation {:<4} Runtime {:<8.4f} fBest: {:<.8f}, fWorst: {:<.8f}, fMean: {:<.8f}, gBest: {:<.2f} ".format(
                g, self.record[1][g] - self.record[0][g], self.record[3][g], self.record[4][g], self.record[5][g],
                self.record[2][g]))

    def reach_best_known_solution(self):
        return False

    def do_selection(self):
        a = np.array(self.pop[2]) / sum(self.pop[2])
        b = np.array([])
        # for i in range(a.shape[0]):
        for i in range(self.pop_size):
            b = np.append(b, sum(a[:i + 1]))
        pop = deepcopy(self.pop)
        # self.pop = [[], [], []]
        for i in range(self.pop_size):
            j = np.argwhere(b > np.random.random())[0, 0]
            # self.pop[0].append(pop[0][j])
            # self.pop[1].append(pop[1][j])
            # self.pop[2].append(pop[2][j])
            self.pop[0][i] = pop[0][j]
            self.pop[1][i] = pop[1][j]
            self.pop[2][i] = pop[2][j]
        self.pop[0][0] = self.best[0]
        self.pop[1][0] = self.best[1]
        self.pop[2][0] = self.best[2]

    def do_init(self, pop=None):
        pass

    def do_crossover(self, i, j):
        pass

    def do_mutation(self, i):
        pass

    def do_evolution(self, pop=None):
        Utils.print("{}Evolution  start{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTYELLOW_EX)
        self.clear()
        self.do_init(pop)
        for g in range(1, self.max_generation + 1):
            if self.reach_best_known_solution():
                break
            self.record[0].append(time.perf_counter())
            self.do_selection()
            for i in range(self.pop_size):
                if self.reach_best_known_solution():
                    break
                # j = np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)[0]
                # rc, rm = self.adaptive_rc_rm_s(i, j)
                # if np.random.random() < rc:
                if np.random.random() < self.rc:
                    j = np.random.choice(np.delete(np.arange(self.pop_size), i), 1, replace=False)[0]
                    self.do_crossover(i, j)
                # if np.random.random() < rm:
                if np.random.random() < self.rm:
                    self.do_mutation(i)
            self.update_t()
            self.record[1].append(time.perf_counter())
            self.show_generation(g)
        Utils.print("{}Evolution finish{}".format("=" * 48, "=" * 48), fore=Utils.fore().LIGHTRED_EX)


class GaNumericOptimization(Ga):
    def __init__(self, pop_size, max_generation, rc, rm, problem, func, max_or_min=0):
        Ga.__init__(self, pop_size, max_generation, rc, rm, problem, func, max_or_min)


class GaTsp(Ga):
    def __init__(self, pop_size, max_generation, rc, rm, problem, func, max_or_min=0):
        Ga.__init__(self, pop_size, max_generation, rc, rm, problem, func, max_or_min)

    def dislocation(self, i):
        code1 = self.pop[0][i].dislocation_operator()
        info = self.problem.decode_ga(self.func, code1)
        # self.pop[0].append(info)
        # self.pop[1].append(info.obj)
        # self.pop[2].append(Utils.calculate_fitness(self.max_or_min, info.obj))
        self.pop[0][i] = self.problem.decode_ga(self.func, code1)
        self.pop[1][i] = self.pop[0][i].obj
        self.pop[2][i] = Utils.calculate_fitness(self.max_or_min, self.pop[1][i])

    def decode_update(self, i, code):
        info = self.problem.decode_ga(self.func, code)
        self.update_individual(i, info.obj, info)

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
            info = self.problem.decode(self.func, code)
            fit = Utils.calculate_fitness(self.max_or_min, info.obj)
            self.pop[0].append(info)
            self.pop[1].append(info.obj)
            self.pop[2].append(fit)
        self.init_best()
        self.record[1].append(time.perf_counter())
        self.show_generation(0)

    def do_crossover(self, i, j):
        func_list = [self.pop[0][i].ga_crossover_pmx, self.pop[0][i].ga_crossover_ox,
                     self.pop[0][i].ga_crossover_heuristic_tsp]
        code1, code2 = self.pop[0][i].ga_crossover_hybrid(self.pop[0][j], func_list)
        self.decode_update(i, code1)
        self.decode_update(j, code2)

    def do_mutation(self, i):
        code1 = self.pop[0][i].ga_mutation_hybrid()
        self.decode_update(i, code1)
