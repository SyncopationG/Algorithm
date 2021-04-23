from ..info import Info
from .scheduling import Scheduling
from ..utils import Utils


class Jsp(Scheduling):
    def __init__(self, low, high, dtype):
        Scheduling.__init__(self, low, high, dtype)

    def decode_operation_based_active(self, func, code, direction=None):
        self.clear()
        if direction not in [0, 1]:
            self.direction = Utils.direction()
        else:
            self.direction = direction
        if self.direction == 1:
            code = code[::-1]
        for g, i in enumerate(code):
            u = self.job[i].nd
            if self.direction == 0:
                j, v = u, u - 1
            else:
                j, v = self.job[i].nop - u - 1, self.job[i].nop - u
            try:
                a = self.job[i].task[v].end
            except KeyError:
                a = 0
            k = self.job[i].task[j].machine
            p = self.job[i].task[j].duration
            for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
                try:
                    early_start = max([a, b])
                except TypeError:
                    early_start = max([0, b])
                if early_start + p <= c:
                    self.job[i].task[j].start = early_start
                    self.job[i].task[j].end = early_start + p
                    self.decode_update_machine_idle(i, j, k, r, self.job[i].task[j].start)
                    self.save_update_decode(i, j, k, g)
                    break
            if self.job[i].task[j].limited_wait is not None:
                if self.direction == 0:
                    index = range(u, -1, -1)
                else:
                    index = range(self.job[i].nop - u - 1, self.job[i].nop, 1)
                while self.constrain_limited_wait(i, index, None) is False:
                    pass
        return Info(self, code, func(self))

    def decode(self, func, code, direction=None):
        a = self.trans_random_key2operation_based(code)
        info = self.decode_operation_based_active(func, a, direction)
        info.code = code
        return info

    def decode_pso(self, func, code, direction=None):
        a = self.trans_random_key2operation_based(code[0])
        info = self.decode_operation_based_active(func, a, direction)
        info.code = code
        return info
