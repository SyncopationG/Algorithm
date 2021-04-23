import os

import chardet
import numpy as np
from colorama import init, Fore

init(autoreset=True)


class Utils:
    @staticmethod
    def create_schedule(shop, n, m, p, tech, proc, w=None, worker=None, best_known=None, limited_wait=None, time_unit=1,
                        low=None, high=None, dtype=None):
        schedule = shop(low, high, dtype)
        schedule.best_known = best_known
        schedule.time_unit = time_unit
        for i in range(m):
            schedule.add_machine(name=i)
        try:
            for i in range(w):
                schedule.add_worker(name=i)
        except TypeError:
            pass
        for i in range(n):
            schedule.add_job(name=i)
            for j in range(p[i]):
                if limited_wait is not None:
                    val_lim = limited_wait[i][j]
                    if val_lim == -1:
                        val_lim = np.inf
                else:
                    val_lim = None
                if worker is not None:
                    val_wok = worker[i][j]
                else:
                    val_wok = None
                schedule.job[i].add_task(tech[i][j], proc[i][j], j, val_lim, val_wok)
        return schedule

    @staticmethod
    def string2data_jsp(string, dtype=int, unit=1):
        try:
            to_data = list(map(dtype, string.split()))
            n, m = int(to_data[0]), int(to_data[1])
            p = [m] * n
            tech = [[] for _ in range(n)]
            proc = [[] for _ in range(n)]
            job, index = 0, 2
            for i in range(n):
                for j in range(m):
                    tech[i].append(int(to_data[index]))
                    proc[i].append(unit * to_data[index + 1])
                    index += 2
            return n, m, p, tech, proc
        except ValueError:
            return None, None, None, None, None

    @staticmethod
    def string2data_fjsp(string, add_one=False, minus_one=True, dtype=int, unit=1):
        try:
            to_data = list(map(dtype, string.split()))
            job, p, tech, prt = 0, [], [], []
            n, m = int(to_data[0]), int(to_data[1])
            index_no, index_nm, index_m, index_t = 2, 3, 4, 5
            while job < n:
                p.append(int(to_data[index_no]))
                tech.append([])
                prt.append([])
                for i in range(p[job]):
                    tech[job].append([])
                    prt[job].append([])
                    int_index_nm = int(to_data[index_nm])
                    for j in range(int_index_nm):
                        int_index_m = int(to_data[index_m])
                        if minus_one is True:
                            tech[job][i].append(int_index_m - 1)
                        elif add_one is True:
                            tech[job][i].append(int_index_m + 1)
                        else:
                            tech[job][i].append(int_index_m)
                        prt[job][i].append(unit * to_data[index_t])
                        index_m += 2
                        index_t += 2
                    index_nm = index_nm + 2 * int_index_nm + 1
                    index_m = index_nm + 1
                    index_t = index_nm + 2
                job += 1
                index_nm = index_nm + 1
                index_m = index_m + 1
                index_t = index_t + 1
                index_no = index_t - 3
            return n, m, p, tech, prt
        except ValueError:
            return None, None, None, None, None

    @staticmethod
    def string2data_drcfjsp(string, add_one=False, minus_one=True, dtype=int, unit=1):
        try:
            a = string.split("\n")
            n, m, w = list(map(int, a[0].split()))
            p = list(map(int, a[1].split()))
            n_job = 0
            tech, worker, proc = [], [], []
            b = [[], [], []]
            for val in a[2:]:
                c, d, e = [], [], []
                data = list(map(dtype, val.split()))
                index_m, index_mn, index_nw, index_wn, index_t = 0, 1, 2, 3, 4
                n_machine = int(data[index_m])
                for k in range(n_machine):
                    if minus_one is True:
                        c.append(int(data[index_mn]) - 1)
                    elif add_one is True:
                        c.append(int(data[index_mn]) + 1)
                    else:
                        c.append(int(data[index_mn]))
                    n_worker = int(data[index_nw])
                    f, g = [], []
                    for cur in range(n_worker):
                        if minus_one is True:
                            f.append(int(data[index_wn]) - 1)
                        elif add_one is True:
                            f.append(int(data[index_wn]) + 1)
                        else:
                            f.append(int(data[index_wn]))
                        g.append(unit * data[index_t])
                        index_wn += 2
                        index_t += 2
                    d.append(f)
                    e.append(g)
                    index_mn = index_t - 1
                    index_nw, index_wn, index_t = index_mn + 1, index_mn + 2, index_mn + 3
                b[0].append(c)
                b[1].append(d)
                b[2].append(e)
                if len(b[0]) == p[n_job]:
                    tech.append(b[0])
                    worker.append(b[1])
                    proc.append(b[2])
                    n_job += 1
                    b = [[], [], []]
            return n, m, w, p, tech, worker, proc
        except ValueError:
            return None, None, None, None, None, None, None

    @staticmethod
    def string2wait(string, nop, dtype=int, unit=1):
        to_data = list(map(dtype, string.split()))
        wait = []
        value = []
        for i, j in enumerate(to_data):
            value.append(j)
            if i + 1 == sum(nop[:len(wait) + 1]):
                wait.append(unit * value)
                value = []
        return wait

    @staticmethod
    def string2data_tsp(string, dtype=int):
        to_data = list(map(dtype, string.split()))
        n = int(to_data[0])
        city = np.array(to_data[1:]).reshape(n, -1).tolist()
        return n, city

    @staticmethod
    def string2data_tsp_cn(string, dtype=int):
        to_data = string.split()
        n = 0
        city = []
        a = []
        for i in to_data:
            try:
                a.append(dtype(i))
            except ValueError:
                a.append(i)
            if len(a) == 3:
                n += 1
                city.append(a)
                a = []
        return n, city

    @staticmethod
    def load_text(file_name):
        try:
            with open(file_name, "rb") as f:
                f_read = f.read()
                f_cha_info = chardet.detect(f_read)
                final_data = f_read.decode(f_cha_info['encoding'])
                return final_data
        except FileNotFoundError:
            return None

    @staticmethod
    def direction():
        return 0 if np.random.random() < 0.5 else 1

    @staticmethod
    def n_time(n_time=None):
        return np.random.choice([1, 2], 1, replace=False)[0] if n_time not in [1, 2] else n_time

    @staticmethod
    def update(max_or_min, obj_old, obj_new):
        return True if (max_or_min == 0 and obj_new > obj_old) or (max_or_min == 1 and obj_new < obj_old) else False

    @staticmethod
    def update_accept_equal(max_or_min, obj_old, obj_new):
        return True if (max_or_min == 0 and obj_new >= obj_old) or (max_or_min == 1 and obj_new <= obj_old) else False

    @staticmethod
    def calculate_fitness(max_or_min, obj):
        return obj if max_or_min == 0 else 1 / (1 + obj)

    @staticmethod
    def fore():
        return Fore

    @staticmethod
    def print(msg, fore=Fore.LIGHTCYAN_EX):
        print(fore + msg)

    @staticmethod
    def make_dir(*args, **kw):
        try:
            os.makedirs(*args, **kw)
        except FileExistsError:
            pass

    @staticmethod
    def save_code_to_txt(file, data):
        if not file.endswith(".txt"):
            file = file + ".txt"
        with open(file, "w", encoding="utf-8") as f:
            f.writelines(str(data))

    @staticmethod
    def save_obj_to_csv(file, data):
        if not file.endswith(".csv"):
            file = file + ".csv"
        with open(file, "w", encoding="utf-8") as f:
            obj, n_iter, direction = [], [], []
            f.writelines("{},{},{},{}\n".format("Test", "Objective", "Iteration", "Direction"))
            for k, v in enumerate(data):
                f.writelines("{},{},{},{}\n".format(k + 1, v[0], v[1] - 1, v[2]))
                obj.append(v[0])
                n_iter.append(v[1] - 1)
                direction.append(v[2])
            f.writelines("{},{}\n".format("MinObj", min(obj)))
            f.writelines("{},{}\n".format("MaxObj", max(obj)))
            f.writelines("{},{:.2f}\n".format("MeanObj", sum(obj) / len(obj)))
            f.writelines("{},{}\n".format("MinIter", min(n_iter)))
            f.writelines("{},{}\n".format("MaxIter", max(n_iter)))
            f.writelines("{},{:.2f}\n".format("MeanIter", sum(n_iter) / len(n_iter)))
            try:
                f.writelines("{},{}\n".format("Direction#0", len(direction) - sum(direction)))
                f.writelines("{},{}\n".format("Direction#1", sum(direction)))
            except TypeError:
                pass

    @staticmethod
    def save_record_to_csv(file, data):
        if not file.endswith(".csv"):
            file = file + ".csv"
        n_row, n_column = len(data[0]), len(data)
        with open(file, "w", encoding="utf-8") as f:
            for i in range(n_row):
                a = ""
                for j in range(n_column):
                    a += "%s," % data[j][i]
                f.writelines(a[:-1] + "\n")
