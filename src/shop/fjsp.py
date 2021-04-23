import numpy as np

from .scheduling import Scheduling
from ..info import Info
from ..utils import Utils


class Fjsp(Scheduling):
    def __init__(self, low, high, dtype):
        Scheduling.__init__(self, low, high, dtype)

    def decode_operation_based_active(self, func, code, mac, wok=None, direction=None):
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
            k = mac[i][j]
            index_k = self.job[i].task[j].machine.index(k)
            if wok is not None:
                w = wok[i][j]
                index = self.job[i].task[j].worker[index_k].index(w)
                p = self.job[i].task[j].duration[index_k][index]
                try:
                    d = max([a, self.worker[w].end])
                except TypeError:
                    if a is not None:
                        d = a
                    else:
                        d = self.worker[w].end
            else:
                w = None
                p = self.job[i].task[j].duration[index_k]
                d = a
            for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
                try:
                    early_start = max([d, b])
                except TypeError:
                    early_start = max([0, b])
                if early_start + p <= c:
                    self.job[i].task[j].start = early_start
                    self.job[i].task[j].end = early_start + p
                    if wok is not None:
                        self.worker[w].start = early_start
                        self.worker[w].end = early_start + p
                    self.decode_update_machine_idle(i, j, k, r, self.job[i].task[j].start)
                    self.save_update_decode(i, j, k, g)
                    break
        return Info(self, code, func(self), mac, wok)

    def decode_only_operation_based_active_worker(self, func, code, direction=None):
        self.clear()
        if direction not in [0, 1]:
            self.direction = Utils.direction()
        else:
            self.direction = direction
        if self.direction == 1:
            code = code[::-1]
        mac = [[None for _ in range(job.nop)] for job in self.job.values()]
        wok = [[None for _ in range(job.nop)] for job in self.job.values()]
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
            start, end, index, duration, worker = [], [], [], [], []
            for k, w_list, p_list in zip(self.job[i].task[j].machine, self.job[i].task[j].worker,
                                         self.job[i].task[j].duration):
                t_start, t_end, t_index = [], [], []
                for w, p in zip(w_list, p_list):
                    try:
                        d = max([a, self.worker[w].end])
                    except TypeError:
                        if a is not None:
                            d = a
                        else:
                            d = self.worker[w].end
                    for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
                        try:
                            early_start = max([d, b])
                        except TypeError:
                            early_start = max([0, b])
                        if early_start + p <= c:
                            res1, res2 = early_start, early_start + p
                            t_start.append(res1)
                            t_end.append(res2)
                            t_index.append(r)
                            break
                index_min_end = np.argwhere(np.array(t_end) == min(t_end))[:, 0]
                duration_in_min_end = np.array([p_list[i] for i in index_min_end])
                choice_min_end_and_duration = np.argwhere(duration_in_min_end == np.min(duration_in_min_end))[:, 0]
                choice = index_min_end[np.random.choice(choice_min_end_and_duration, 1, replace=False)[0]]
                start.append(t_start[choice])
                end.append(t_end[choice])
                index.append(t_index[choice])
                duration.append(p_list[choice])
                worker.append(w_list[choice])
            index_min_end = np.argwhere(np.array(end) == min(end))[:, 0]
            duration_in_min_end = np.array([duration[i] for i in index_min_end])
            choice_min_end_and_duration = np.argwhere(duration_in_min_end == np.min(duration_in_min_end))[:, 0]
            choice = index_min_end[np.random.choice(choice_min_end_and_duration, 1, replace=False)[0]]
            k, w, p, r = self.job[i].task[j].machine[choice], worker[choice], duration[choice], index[choice]
            mac[i][j], wok[i][j] = k, w
            self.job[i].task[j].start = self.worker[w].start = start[choice]
            self.job[i].task[j].end = self.worker[w].end = end[choice]
            self.decode_update_machine_idle(i, j, k, r, start[choice])
            self.save_update_decode(i, j, k, g)
        return Info(self, code, func(self), mac, wok)

    def decode_only_operation_based_active(self, func, code, direction=None):
        self.clear()
        if direction not in [0, 1]:
            self.direction = Utils.direction()
        else:
            self.direction = direction
        if self.direction == 1:
            code = code[::-1]
        mac = [[None for _ in range(job.nop)] for job in self.job.values()]
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
            start, end, index, duration = [], [], [], []
            for k, p in zip(self.job[i].task[j].machine, self.job[i].task[j].duration):
                for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
                    try:
                        early_start = max([a, b])
                    except TypeError:
                        early_start = max([0, b])
                    if early_start + p <= c:
                        res1, res2 = early_start, early_start + p
                        start.append(res1)
                        end.append(res2)
                        index.append(r)
                        duration.append(p)
                        break
            index_min_end = np.argwhere(np.array(end) == min(end))[:, 0]
            duration_in_min_end = np.array([duration[i] for i in index_min_end])
            choice_min_end_and_duration = np.argwhere(duration_in_min_end == np.min(duration_in_min_end))[:, 0]
            choice = index_min_end[np.random.choice(choice_min_end_and_duration, 1, replace=False)[0]]
            k, p, r = self.job[i].task[j].machine[choice], duration[choice], index[choice]
            mac[i][j] = k
            self.job[i].task[j].start = start[choice]
            self.job[i].task[j].end = end[choice]
            self.decode_update_machine_idle(i, j, k, r, start[choice])
            self.save_update_decode(i, j, k, g)
            if self.job[i].task[j].limited_wait is not None:
                if self.direction == 0:
                    index = range(u, -1, -1)
                else:
                    index = range(self.job[i].nop - u - 1, self.job[i].nop, 1)
                while self.constrain_limited_wait(i, index, mac) is False:
                    pass
        return Info(self, code, func(self), mac)

    def decode(self, func, code, direction=None):
        a = self.trans_random_key2operation_based(code)
        info = self.decode_only_operation_based_active(func, a, direction)
        info.code = code
        return info

    def decode_pso(self, func, code, direction=None):
        a = self.trans_random_key2operation_based(code[0])
        info = self.decode_only_operation_based_active(func, a, direction)
        info.code = code
        return info

    def decode_worker(self, func, code, direction=None):
        a = self.trans_random_key2operation_based(code)
        info = self.decode_only_operation_based_active_worker(func, a, direction)
        info.code = code
        return info

    def decode_worker_pso(self, func, code, direction=None):
        a = self.trans_random_key2operation_based(code[0])
        info = self.decode_only_operation_based_active_worker(func, a, direction)
        info.code = code
        return info
