import copy

import numpy as np

from ..info import Info
from .scheduling import Scheduling

deepcopy = copy.deepcopy


class Hfsp(Scheduling):
    def __init__(self, low, high, dtype):
        Scheduling.__init__(self, low, high, dtype)

    def decode_permutation(self, func, code):
        self.clear()
        copy_code = deepcopy(code)
        mac, j = [[None for _ in range(job.nop)] for job in self.job.values()], 0
        while self.any_task_not_done():
            for i in copy_code:
                try:
                    a = self.job[i].task[j - 1].end
                except KeyError:
                    a = 0
                start, end, duration = [], [], []
                for k, p in zip(self.job[i].task[j].machine, self.job[i].task[j].duration):
                    early_start = max([a, self.machine[k].end])
                    start.append(early_start)
                    end.append(early_start + p)
                    duration.append(p)
                index_min_end = np.argwhere(np.array(end) == min(end))[:, 0]
                duration_in_min_end = np.array([duration[i] for i in index_min_end])
                choice_min_end_and_duration = np.argwhere(duration_in_min_end == np.min(duration_in_min_end))[:, 0]
                choice = index_min_end[np.random.choice(choice_min_end_and_duration, 1, replace=False)[0]]
                k, p = self.job[i].task[j].machine[choice], duration[choice]
                mac[i][j] = k
                self.job[i].task[j].start = start[choice]
                self.job[i].task[j].end = end[choice]
                if self.machine[k].end < self.job[i].task[j].end:
                    self.machine[k].end = self.job[i].task[j].end
                if self.job[i].task[0].limited_wait is not None:
                    while self.constrain_limited_wait(i, range(j, -1, -1), mac) is False:
                        pass
            copy_code = copy_code[np.argsort([self.job[i].task[j].start for i in copy_code])]
            j += 1
        return Info(self, code, func(self))

    def decode(self, func, code):
        info = self.decode_permutation(func, np.argsort(code))
        info.code = code
        return info

    def decode_pso(self, func, code):
        info = self.decode_permutation(func, np.argsort(code[0]))
        info.code = code
        return info
