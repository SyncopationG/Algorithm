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
