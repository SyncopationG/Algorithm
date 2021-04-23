import copy

import numpy as np

from ..info import Info
from .scheduling import Scheduling

deepcopy = copy.deepcopy


class Fsp(Scheduling):
    def __init__(self, low, high, dtype):
        Scheduling.__init__(self, low, high, dtype)

    def decode_permutation(self, func, code):
        self.clear()
        for i in code:
            for j in range(self.job[i].nop):
                k = self.job[i].task[j].machine
                p = self.job[i].task[j].duration
                try:
                    a = self.job[i].task[j - 1].end
                except KeyError:
                    a = 0
                self.job[i].task[j].start = max([a, self.machine[k].end])
                self.job[i].task[j].end = self.job[i].task[j].start + p
                if self.machine[k].end < self.job[i].task[j].end:
                    self.machine[k].end = self.job[i].task[j].end
            if self.job[i].task[0].limited_wait is not None:
                for j_end2head in range(self.job[i].nop - 1, 0, -1):
                    limited_wait = self.job[i].task[j_end2head - 1].limited_wait
                    end = self.job[i].task[j_end2head - 1].end
                    start = self.job[i].task[j_end2head].start
                    if start - end - limited_wait > 0:
                        k = self.job[i].task[j_end2head - 1].machine
                        self.job[i].task[j_end2head - 1].start = start - self.job[i].task[j_end2head - 1].duration
                        self.job[i].task[j_end2head - 1].end = start
                        if self.machine[k].end < self.job[i].task[j_end2head - 1].end:
                            self.machine[k].end = self.job[i].task[j_end2head - 1].end
        return Info(self, code, func(self))

    def decode(self, func, code):
        info = self.decode_permutation(func, np.argsort(code))
        info.code = code
        return info

    def decode_pso(self, func, code):
        info = self.decode_permutation(func, np.argsort(code[0]))
        info.code = code
        return info
