import copy

import numpy as np
from .define import Name, Operator
from .utils import Utils

deepcopy = copy.deepcopy


class Info:
    def __init__(self, problem, code, obj):
        self.problem = deepcopy(problem)
        self.code = code
        self.obj = obj

    def print(self):
        Utils.print("%s %s" % (self.code, self.obj), fore=Utils.fore().LIGHTYELLOW_EX)

    def save(self, file):
        if not file.endswith(".txt"):
            file += ".txt"
        a = "%s %s\n" % (self.code, self.obj)
        with open(file, "w", encoding="utf-8") as f:
            for i, j in enumerate(a):
                f.writelines(j)
                if (i + 1) % 100 == 0:
                    f.writelines("\n")

    def repair(self, code):
        for i, (j, k) in enumerate(zip(code, self.problem.dtype)):
            if self.problem.low[i] > j:
                code[i] = self.problem.low[i]
            if self.problem.high[i] < j:
                code[i] = self.problem.high[i]
            code[i] = k(code[i])
        return code

    def de_mutation(self, f, info2, info3, info4, info5, info_best):
        if self.problem.operator[Name.de] in [Operator.default, Operator.de_rand1]:
            return self.de_mutation_sequence_rand1(f, info2, info3)
        elif self.problem.operator[Name.de] == Operator.de_best1:
            return self.de_mutation_sequence_best1(f, info2, info_best)
        elif self.problem.operator[Name.de] == Operator.de_c2best1:
            return self.de_mutation_sequence_c2best1(f, info2, info_best)
        elif self.problem.operator[Name.de] == Operator.de_best2:
            return self.de_mutation_sequence_best2(f, info2, info3, info4, info_best)
        return self.de_mutation_sequence_rand2(f, info2, info3, info4, info5)

    def de_crossover(self, cr, info2):
        code1 = deepcopy(self.code)
        code2 = deepcopy(info2.code)
        for i, (j, k) in enumerate(zip(code1, code2)):
            if np.random.random() < cr:
                code1[i], code2[i] = k, j
        return code1, code2

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

    def jaya_update(self, best, worst, rand):
        if self.problem.operator[Name.jaya] in [Operator.default, Operator.jaya_classic]:
            return self.jaya_classic(best, worst)
        return self.jaya_rand(best, worst, rand)

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

    def pso_update(self, c1, c2, w, p_best, g_best):
        if self.problem.operator[Name.pso] in [Operator.default, Operator.pso_classic]:
            return self.pso_classic(c1, c2, w, p_best, g_best)
        return self.pso_classic(c1, c2, w, p_best, g_best)

    def pso_classic(self, c1, c2, w, p_best, g_best):
        a, b = np.random.random(2)
        c = w * self.code[1] + c1 * a * (p_best.code[0] - self.code[0]) + c2 * b * (g_best.code[0] - self.code[0])
        d = self.code[1] + c
        return self.repair(c), self.repair(d)

    def sa_update(self, t):
        if self.problem.operator[Name.sa] in [Operator.default, Operator.sa_classic]:
            return self.sa_classic(t)
        return self.sa_classic(t)

    def sa_classic(self, t):
        a = -self.problem.var_range + 2 * self.problem.var_range * np.random.random(self.problem.n)
        b = a / np.sqrt(sum([i ** 2 for i in a]))
        new = self.code + b * t
        return self.repair(new)

    def ga_crossover(self, info):
        if self.problem.operator[Name.ga_x] in [Operator.default, Operator.ga_x_pmx]:
            return self.ga_crossover_pmx(info)
        elif self.problem.operator[Name.ga_x] == Operator.ga_x_ox:
            return self.ga_crossover_ox(info)
        return self.ga_crossover_heuristic_tsp(info)

    def ga_mutation(self):
        if self.problem.operator[Name.ga_m] in [Operator.default, Operator.ga_m_tpe]:
            return self.ga_mutation_tpe()
        elif self.problem.operator[Name.ga_m] == Operator.ga_m_insert:
            return self.ga_mutation_insert()
        return self.ga_mutation_sr()

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

    def ga_crossover_heuristic_tsp(self, info):
        code1, code1c = deepcopy(self.code), deepcopy(self.code)
        code2, code2c = deepcopy(info.code), deepcopy(info.code)
        a = b = np.random.randint(0, self.problem.n, 1)[0]
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

    def ga_mutation_tpe(self):
        code = deepcopy(self.code)
        a = np.random.choice(range(self.problem.n), 2, replace=False)
        code[a] = code[a[::-1]]
        return code

    def ga_mutation_insert(self):
        code = deepcopy(self.code)
        try:
            b, c = np.random.choice(range(self.problem.n), 2, replace=False)
            if b > c:
                b, c = c, b
            val = code[c]
            obj = np.delete(code, c)
            code = np.insert(obj, b, val)
        except ValueError:
            code = code[::-1]
        return code

    def ga_mutation_sr(self):
        code = deepcopy(self.code)
        a, b = np.random.choice(range(self.problem.n), 2, replace=False)
        if a > b:
            a, b = b, a
        c = range(a, b + 1)
        code[c] = code[c[::-1]]
        return code

    @staticmethod
    def do_tabu_search(code, i, j, w):
        if i > j:
            i, j = j, i
        if w == 0:
            obj = np.delete(code, j)
            code = np.insert(obj, i, code[j])
        elif w == 1:
            obj = np.delete(code, i)
            code = np.insert(obj, j - 1, code[i])
            code[j], code[j - 1] = code[j - 1], code[j]
        else:
            code[i], code[j] = code[j], code[i]
        return code

    def ts_permutation(self, tabu_list, max_tabu):
        code = deepcopy(self.code)
        n_try = 0
        while n_try < max_tabu:
            n_try += 1
            try:
                i, j = np.random.choice(self.problem.n, 2, replace=False)
                w = np.random.choice(range(3), 1, replace=False)[0]
                tabu = {"way-%s" % w, i, j}
                if tabu not in tabu_list:
                    tabu_list.append(tabu)
                    code = self.do_tabu_search(code, i, j, w)
                    break
            except ValueError:
                pass
        return code

    def dislocation_operator(self, direction=0):
        return np.hstack([self.code[1:], self.code[0]]) if direction == 0 else np.hstack(
            [self.code[-1], self.code[:-1]])
