import os

import numpy as np
from colorama import init, Fore

init(autoreset=True)


class Utils:
    @staticmethod
    def similarity(a, b):
        return sum([1 if i == j else 0 for i, j in zip(a, b)]) / len(a)

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
    def len_tabu(n):
        return 10 * n

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
