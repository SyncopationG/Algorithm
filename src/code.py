import numpy as np


class Code:
    @staticmethod
    def code(low, high, dtype):
        a = np.array([])
        for i, j, k in zip(low, high, dtype):
            b = i + (j - i) * np.random.random()
            a = np.append(a, k(b))
        return a

    @staticmethod
    def code_particle(low, high, dtype):
        a, b = np.array([]), np.array([])  # a：位置, b：速度
        for i, j, k in zip(low, high, dtype):
            c = i + (j - i) * np.random.random()
            a, b = np.append(a, k(c)), np.append(b, 0)
        return a, b

    @staticmethod
    def code_permutation(length):
        return np.random.permutation(length)

    @staticmethod
    def code_operation_sequence(p):
        a = np.array([], dtype=int)
        for i, j in enumerate(p):
            a = np.append(a, [i] * j)
        np.random.shuffle(a)
        return a

    @staticmethod
    def code_machine_assignment(n, p, tech):
        a = []
        for i in range(n):
            a.append([])
            for j in range(p[i]):
                a[i].append(np.random.choice(tech[i][j], 1, replace=False)[0])
        return a

    @staticmethod
    def code_machine_assignment_bml(n, m, p, tech, proc):
        a, b = [], [0] * m
        for i in range(n):
            a.append([])
            for j in range(p[i]):
                c = []
                for m, d in zip(tech[i][j], proc[i][j]):
                    c.append(b[m] + d)
                index = c.index(min(c))
                k = tech[i][j][index]
                b[k] += proc[i][j][index]
                a[i].append(k)
        return a

    @staticmethod
    def code_machine_assignment_bml_worker(n, m, p, tech, proc):
        a, b = [], [0] * m
        for i in range(n):
            a.append([])
            for j in range(p[i]):
                c = []
                proc_mean = np.mean(proc[i][j])
                for u in tech[i][j]:
                    c.append(b[u] + proc_mean)
                index = c.index(min(c))
                k = tech[i][j][index]
                b[k] += proc_mean
                a[i].append(k)
        return a

    @staticmethod
    def code_worker_assignment(n, p, tech, worker, mac):
        a = []
        for i in range(n):
            a.append([])
            for j in range(p[i]):
                index = tech[i][j].index(mac[i][j])
                a[i].append(np.random.choice(worker[i][j][index], 1, replace=False)[0])
        return a

    @staticmethod
    def code_worker_assignment_bwl(n, w, p, tech, worker, proc, mac):
        a, b = [], [0] * w
        for i in range(n):
            a.append([])
            for j in range(p[i]):
                index = tech[i][j].index(mac[i][j])
                c = []
                for u, v in zip(worker[i][j][index], proc[i][j][index]):
                    c.append(b[u] + v)
                index_c = c.index(min(c))
                d = worker[i][j][index][index_c]
                b[d] += proc[i][j][index][index_c]
                a[i].append(d)
        return a
