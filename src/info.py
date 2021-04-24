import copy
import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from .utils import Utils

deepcopy = copy.deepcopy
dt = datetime.datetime
tmdelta = datetime.timedelta
COLORS = list(mcolors.CSS4_COLORS)
COLORS_REMOVE = ['black', "white"]
COLORS_REMOVE.extend([i for i in COLORS if i.startswith('dark')])
COLORS_REMOVE.extend([i for i in COLORS if i.startswith('light')])
[COLORS.remove(i) for i in COLORS_REMOVE]
[COLORS.pop(j - i) for i, j in enumerate(range(11))]
[COLORS.pop(j - i) for i, j in enumerate([6, ])]
LEN_COLORS = len(COLORS)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class GanttChart:
    def __init__(self, problem=None, mac=None, wok=None):
        self.problem = deepcopy(problem)
        self.mac = mac
        self.wok = wok

    def ganttChart_png(self, filename="GanttChart", fig_width=9, fig_height=5, lang=0,
                       random_colors=False, with_operation=True, with_start_end=False,
                       height=0.8, scale_more=None, text_rotation=1, x_step=None, dpi=200, show=False, ):
        if random_colors:
            random.shuffle(COLORS)
        plt.figure(figsize=[fig_width, fig_height])
        plt.yticks(range(self.problem.m), range(1, self.problem.m + 1))
        plt.xticks([], [])
        ax = plt.gca()
        scale_more = 12 if scale_more is None else scale_more
        x_step = int(self.problem.makespan // 10) if x_step is None else x_step
        for job in self.problem.job.values():
            for task in job.task.values():
                if self.mac is None:
                    machine = task.machine
                else:
                    machine = self.mac[job.index][task.index]
                if self.wok is None:
                    index_color = job.index
                else:
                    index_color = self.wok[job.index][task.index]
                width = task.end - task.start
                left = [task.start, self.problem.makespan - task.end][self.problem.direction]
                plt.barh(
                    y=machine, width=width,
                    left=left, color=COLORS[(index_color + 1) % LEN_COLORS],
                    edgecolor="black", linewidth=0.5,
                )
                if with_operation:
                    mark = r"$O_{%s,%s}$" % (job.index + 1, task.index + 1)
                    plt.text(
                        x=left + 0.5 * width, y=machine,
                        s=mark, c="black",
                        ha="center", va="center", rotation="vertical",
                    )
                if with_start_end:
                    if self.problem.direction == 0:
                        val = [task.start, task.end]
                    else:
                        val = [self.problem.makespan - task.end, self.problem.makespan - task.start]
                    for x in val:
                        s = r"$_{%s}$" % int(x)
                        rotation = text_rotation
                        if text_rotation in [0, 1]:
                            rotation = ["horizontal", "vertical"][text_rotation]
                        plt.text(
                            x=x, y=machine - height * 0.5,
                            s=s, c="black",
                            ha="center", va="top",
                            rotation=rotation,
                        )
        if not with_operation and self.wok is None:
            for job in self.problem.job.values():
                plt.barh(0, 0, color=COLORS[(job.index + 1) % LEN_COLORS], label=job.index + 1)
            plt.barh(y=0, width=self.problem.makespan / scale_more, left=self.problem.makespan, color="white")
            if lang == 0:
                title = r"${Job}$"
            else:
                title = "工件"
            plt.legend(loc="best", title=title)
        if not with_start_end:
            ymin = -0.5
            ymax = self.problem.m + ymin
            plt.vlines(self.problem.makespan, ymin, ymax, colors="red", linestyles="--")
            plt.text(self.problem.makespan, ymin, "{}".format(int(self.problem.makespan / self.problem.time_unit)))
            x_ticks = range(0, self.problem.makespan + x_step, x_step)
            plt.xticks(x_ticks, x_ticks)
            [ax.spines[name].set_color('none') for name in ["top", "right"]]
        else:
            [ax.spines[name].set_color('none') for name in ["top", "right", "bottom", "left"]]
        if lang == 0:
            plt.ylabel(r"${Machine}$")
            plt.xlabel(r"${Time}$")
        else:
            plt.ylabel("机器")
            plt.xlabel("时间")
        if self.wok is not None:
            for worker in self.problem.worker.values():
                plt.barh(0, 0, color=COLORS[(worker.index + 1) % LEN_COLORS], label=worker.index + 1)
            plt.barh(y=0, width=self.problem.makespan / scale_more, left=self.problem.makespan, color="white")
            if lang == 0:
                title = r"${Worker}$"
            else:
                title = "工人"
            plt.legend(loc="best", title=title)
        plt.margins()
        plt.tight_layout()
        plt.gcf().subplots_adjust(left=0.08, bottom=0.12)
        plt.savefig("{}.png".format(filename), dpi=dpi)
        if show:
            plt.show()
        plt.clf()
        Utils.print("Create {}.png".format(filename), fore=Utils.fore().LIGHTCYAN_EX)


class Info(GanttChart):
    def __init__(self, problem, code, obj, mac=None, wok=None):
        GanttChart.__init__(self, problem, mac, wok)
        self.code = code
        self.obj = obj

    def repair(self, code):
        for i, (j, k) in enumerate(zip(code, self.problem.dtype)):
            if self.problem.low[i] > j:
                code[i] = self.problem.low[i]
            if self.problem.high[i] < j:
                code[i] = self.problem.high[i]
            code[i] = k(code[i])
        return code

    def de_mutation_sequence_rand1(self, f, info2, info3):
        code1 = self.code
        code2 = info2.code
        code3 = info3.code
        new = code1 + f * (code2 - code3)
        return self.repair(new)

    def de_mutation_sequence_best1(self, f, info2, info_best):
        code1 = self.code
        code2 = info2.code
        code_best = info_best.code
        new = code_best + f * (code1 - code2)
        return self.repair(new)

    def de_mutation_sequence_c2best1(self, f, info2, info_best):
        code1 = self.code
        code2 = info2.code
        code_best = info_best.code
        new = code1 + f * (code_best - code1) + f * (code1 - code2)
        return self.repair(new)

    def de_mutation_sequence_best2(self, f, info2, info3, info4, info_best):
        code1 = self.code
        code2 = info2.code
        code3 = info3.code
        code4 = info4.code
        code_best = info_best.code
        new = code_best + f * (code1 - code2) + f * (code3 - code4)
        return self.repair(new)

    def de_mutation_sequence_rand2(self, f, info2, info3, info4, info5):
        code1 = self.code
        code2 = info2.code
        code3 = info3.code
        code4 = info4.code
        code5 = info5.code
        new = code1 + f * (code2 - code3) + f * (code4 - code5)
        return self.repair(new)

    def de_mutation_sequence_hybrid(self, f, info2, info3, info4, info5, info_best):
        a = np.random.random()
        if a < 0.2:
            return self.de_mutation_sequence_rand1(f, info2, info3)
        elif a < 0.4:
            return self.de_mutation_sequence_best1(f, info2, info_best)
        elif a < 0.6:
            return self.de_mutation_sequence_c2best1(f, info2, info_best)
        elif a < 0.8:
            return self.de_mutation_sequence_best2(f, info2, info3, info4, info_best)
        return self.de_mutation_sequence_rand2(f, info2, info3, info4, info5)

    def de_crossover_sequence_normal(self, cr, info2):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info2.code)
        for i, (j, k) in enumerate(zip(code1, code2)):
            if np.random.random() < cr:
                code1[i], code2[i] = k, j
        return code1, code2

    def jaya_classic(self, best, worst):
        a, b = np.random.random(2)
        code = np.abs(self.code)
        new = self.code + a * (best.code - code) - b * (worst.code - code)
        return self.repair(new)

    def jaya_rand(self, best, worst, rand):
        a, b, c = np.random.random(3)
        code = np.abs(self.code)
        new = self.code + a * (best.code - code) - b * (worst.code - code) + c * (rand.code - code)
        return self.repair(new)

    def jaya_hybrid(self, best, worst, rand):
        return self.jaya_classic(best, worst) if np.random.random() < 0.5 else self.jaya_rand(best, worst, rand)

    def pso_classic(self, c1, c2, w, p_best, g_best):
        a, b = np.random.random(2)
        c = w * self.code[1] + c1 * a * (p_best.code[0] - self.code[0]) + c2 * b * (g_best.code[0] - self.code[0])
        d = self.code[1] + c
        return self.repair(c), self.repair(d)

    def sa_classic(self, t):
        a = -self.problem.var_range + 2 * self.problem.var_range * np.random.random(self.problem.n_dim)
        b = a / np.sqrt(sum([i ** 2 for i in a]))
        new = self.code + b * t
        return self.repair(new)

    def similarity(self, info):
        return 1 - np.count_nonzero(self.code - info.code)

    def ga_crossover_pmx(self, info):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        a, b = np.random.choice(self.problem.n, 2, replace=False)
        if a > b:
            a, b = b, a
        r_a_b = range(a, b)
        r_left = np.delete(range(self.problem.n), r_a_b)
        middle_1, middle_2 = code1[r_a_b], code2[r_a_b]
        left_1, left_2 = code1[r_left], code2[r_left]
        code1[r_a_b], code2[r_a_b] = middle_2, middle_1
        mapping = [[], []]
        for i, j in zip(middle_1, middle_2):
            if j in middle_1 and i not in middle_2:
                index = np.argwhere(middle_1 == j)[0, 0]
                value = middle_2[index]
                while True:
                    if value in middle_1:
                        index = np.argwhere(middle_1 == value)[0, 0]
                        value = middle_2[index]
                    else:
                        break
                mapping[0].append(i)
                mapping[1].append(value)
            elif j not in middle_1 and i not in middle_2:
                mapping[0].append(i)
                mapping[1].append(j)
        for i, j in zip(mapping[0], mapping[1]):
            if i in left_1:
                left_1[np.argwhere(left_1 == i)[0, 0]] = j
            elif i in left_2:
                left_2[np.argwhere(left_2 == i)[0, 0]] = j
            if j in left_1:
                left_1[np.argwhere(left_1 == j)[0, 0]] = i
            elif j in left_2:
                left_2[np.argwhere(left_2 == j)[0, 0]] = i
        code1[r_left], code2[r_left] = left_1, left_2
        return code1, code2

    def ga_crossover_ox(self, info):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info.code)
        a, b = np.random.choice(range(1, self.problem.n - 1), 2, replace=False)
        if a > b:
            a, b = b, a
        r_a_b = range(a, b + 1)
        left_a, right_b = range(a), range(b + 1, self.problem.n)
        left_b_a = np.hstack([right_b, left_a])
        middle1, middle2 = code1[r_a_b], code2[r_a_b]
        left1, left2 = code1[left_a], code2[left_a]
        right1, right2 = code1[right_b], code2[right_b]
        cycle1, cycle2 = np.hstack([right1, left1, middle1]), np.hstack([right2, left2, middle2])
        change1, change2 = [], []
        for i, j in zip(cycle1, cycle2):
            if j not in middle1:
                change1.append(j)
            if i not in middle2:
                change2.append(i)
        code1[left_b_a], code2[left_b_a] = change1, change2
        return code1, code2

    def ga_crossover_heuristic_tsp(self, info, c=None):
        code1, code1c = deepcopy(self.code), deepcopy(self.code)
        code2, code2c = deepcopy(info.code), deepcopy(info.code)
        a = b = np.random.randint(0, self.problem.n, 1)[0] if c is None else c
        res1, res2 = np.array([a, ], dtype=int), np.array([a, ], dtype=int)
        while res1.shape[0] < self.problem.n:  # child1
            index1_a, index2_a = np.argwhere(code1 == a)[0, 0], np.argwhere(code2 == a)[0, 0]
            index1_a_next, index2_a_next = (index1_a + 1) % code1.shape[0], (index2_a + 1) % code2.shape[0]
            c1, c2 = code1[index1_a_next], code2[index2_a_next]
            a = c1 if self.problem.distance(a, c1) <= self.problem.distance(a, c2) else c2
            code1, code2 = np.delete(code1, index1_a), np.delete(code2, index2_a)
            res1 = np.append(res1, a)
        while res2.shape[0] < self.problem.n:  # child2
            index1_b, index2_b = np.argwhere(code1c == b)[0, 0], np.argwhere(code2c == b)[0, 0]
            index1_b_pre, index2_b_pre = index1_b - 1, index2_b - 1
            c1, c2 = code1c[index1_b_pre], code2c[index2_b_pre]
            b = c1 if self.problem.distance(b, c1) <= self.problem.distance(b, c2) else c2
            code1c, code2c = np.delete(code1c, index1_b), np.delete(code2c, index2_b)
            res2 = np.append(res2, b)
        return res1, res2

    def ga_crossover_hybrid(self, info, func_list=None):
        if func_list is None:
            func_list = [self.ga_crossover_pmx, self.ga_crossover_ox]
        func = np.random.choice(func_list, 1, replace=False)[0]
        return func(info)

    def ga_mutation_tpe(self, n_time=None):
        code = deepcopy(self.code)
        for n_do in range(Utils.n_time(n_time)):
            a = np.random.choice(range(self.problem.n), 2, replace=False)
            code[a] = code[a[::-1]]
        return code

    def ga_mutation_insert(self, n_time=None):
        code = deepcopy(self.code)
        try:
            for n_do in range(Utils.n_time(n_time)):
                b, c = np.random.choice(range(self.problem.n), 2, replace=False)
                if b > c:
                    b, c = c, b
                val = code[c]
                obj = np.delete(code, c)
                code = np.insert(obj, b, val)
        except ValueError:
            code = code[::-1]
        return code

    def ga_mutation_reverse(self, n_time=None):
        code = deepcopy(self.code)
        for n_do in range(Utils.n_time(n_time)):
            a, b = np.random.choice(range(self.problem.n), 2, replace=False)
            if a > b:
                a, b = b, a
            c = range(a, b + 1)
            code[c] = code[c[::-1]]
        return code

    def ga_mutation_hybrid(self, func_list=None):
        if func_list is None:
            func_list = [self.ga_mutation_tpe, self.ga_mutation_insert, self.ga_mutation_reverse]
        func = np.random.choice(func_list, 1, replace=False)[0]
        return func()

    def dislocation_operator(self):
        return np.hstack([self.code[1:], self.code[0]])
