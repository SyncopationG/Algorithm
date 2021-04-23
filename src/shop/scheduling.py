import numpy as np

from ..code import Code
from ..resource.job import Job
from ..resource.machine import Machine
from ..resource.worker import Worker


class Scheduling(Code):  # 调度资源
    def __init__(self, low, high, dtype):
        self.low = low
        self.high = high
        self.dtype = dtype
        try:
            self.n_dim = len(low)
            self.var_range = np.array([i - j for i, j in zip(high, low)])
        except TypeError:
            pass
        self.job = {}  # 工件
        self.machine = {}  # 机器
        self.worker = {}  # 工人
        self.best_known = None  # 已知下界值
        self.time_unit = 1  # 加工时间单位
        self.direction = 0  # 解码用：正向时间表、反向时间表（仅Jsp, Fjsp）

    def clear(self):  # 解码前要进行清空, 方便快速地进行下一次解码
        for i in self.job.keys():
            self.job[i].clear()
        for i in self.machine.keys():
            self.machine[i].clear()
        for i in self.worker.keys():
            self.worker[i].clear()
        self.direction = 0

    @property
    def n(self):  # 工件数量
        return len(self.job)

    @property
    def m(self):  # 机器数量
        return len(self.machine)

    @property
    def w(self):  # 工人数量
        return len(self.worker)

    @property
    def length(self):  # 总的工序数量
        return sum([job.nop for job in self.job.values()])

    @property
    def makespan(self):  # 工期
        return max([machine.end for machine in self.machine.values()])

    @property
    def a_operation_based_code(self):
        a = np.array([], dtype=int)
        for i in self.job.keys():
            a = np.append(a, [i] * self.job[i].nop)
        return a

    def trans_random_key2operation_based(self, code):  # 转换编码：基于随机键的编码->基于工序的编码
        return self.a_operation_based_code[np.argsort(code)]

    def any_task_not_done(self):  # 解码用：判断是否解码结束
        return any([any([task.start is None for task in job.task.values()]) for job in self.job.values()])

    def add_machine(self, name=None, index=None):  # 添加机器
        if index is None:
            index = self.m
        self.machine[index] = Machine(index, name)

    def add_job(self, due_date=None, name=None, index=None):  # 添加工件
        if index is None:
            index = self.n
        self.job[index] = Job(index, due_date, name)

    def add_worker(self, name=None, index=None):  # 添加工人
        if index is None:
            index = self.w
        self.worker[index] = Worker(index, name)

    def save_update_decode(self, i, j, k, g):  # 解码用: 更新、保存解码信息
        self.job[i].nd += 1
        self.job[i].index_list[j] = g
        self.machine[k].index_list.append(g)

    def decode_update_machine_idle(self, i, j, k, r, early_start):  # 解码用：更新机器空闲时间
        if self.machine[k].idle[1][r] - self.job[i].task[j].end > 0:  # 添加空闲时间段
            self.machine[k].idle[0].insert(r + 1, self.job[i].task[j].end)
            self.machine[k].idle[1].insert(r + 1, self.machine[k].idle[1][r])
        if self.machine[k].idle[0][r] == early_start:  # 删除空闲时间段
            self.machine[k].idle[0].pop(r)
            self.machine[k].idle[1].pop(r)
        else:
            self.machine[k].idle[1][r] = early_start  # 更新空闲时间段
        if self.machine[k].end < self.job[i].task[j].end:  # 更新机器上的最大完工时间
            self.machine[k].end = self.job[i].task[j].end

    def constrain_limited_wait_release_job(self, i, j, k, p):  # 等待时间有限约束：释放工件占用机器的时间
        try:  # 工件紧前是机器的空闲时间段
            index_pre = self.machine[k].idle[1].index(self.job[i].task[j].start)
        except ValueError:  # 工件紧前的是工件
            index_pre = None
        if self.job[i].task[j].end == self.machine[k].end:  # 工件是机器上的当前最后一个加工的
            try:  # 工件紧前是机器的空闲时间段, 更新机器上的最大完工时间, 删除此空闲时间段
                self.machine[k].end = self.machine[k].idle[0][index_pre]
                self.machine[k].idle[0].pop(index_pre)
                self.machine[k].idle[1].pop(index_pre)
            except TypeError:  # 工件紧前是工件, 更新机器上的最大完工时间
                self.machine[k].end -= p
            self.machine[k].idle[0][-1] = self.machine[k].end
        else:  # 工件不是机器上的当前最后一个加工的
            try:  # 工件紧后是机器的空闲时间段
                index_next = self.machine[k].idle[0].index(self.job[i].task[j].end)
            except ValueError:  # 工件紧后是工件
                index_next = None
            if index_pre is None and index_next is None:  # 工件紧前、紧后都是工件
                self.machine[k].idle[0].append(self.job[i].task[j].start)
                self.machine[k].idle[1].append(self.job[i].task[j].end)
                self.machine[k].idle[0].sort()
                self.machine[k].idle[1].sort()
            elif index_pre is not None and index_next is not None:  # 工件紧前、紧后都是机器的空闲时间段
                # 更新紧前空闲时间段的空闲结束开始为紧后空闲时间段的空闲结束时刻
                self.machine[k].idle[1][index_pre] = self.machine[k].idle[1][index_next]
                self.machine[k].idle[0].pop(index_next)
                self.machine[k].idle[1].pop(index_next)
            elif index_pre is not None and index_next is None:  # 工件紧前是机器的空闲时间段, 紧后是工件
                self.machine[k].idle[1][index_pre] = self.job[i].task[j].end
            else:  # 工件紧前是工件, 紧后是机器的空闲时间段
                self.machine[k].idle[0][index_next] = self.job[i].task[j].start

    def constrain_limited_wait_repair_interval(self, i, index, cursor, mac=None):  # 等待时间有限约束：合法化加工时间间隔
        for j, j_pre in zip(index[:cursor + 1][::-1], index[1:cursor + 2][::-1]):
            if mac is None:
                k = self.job[i].task[j].machine
                p = self.job[i].task[j].duration
            else:
                k = mac[i][j]
                p = self.job[i].task[j].duration[self.job[i].task[j].machine.index(k)]
            self.constrain_limited_wait_release_job(i, j, k, p)
            for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
                early_start = max([self.job[i].task[j_pre].end, b])
                if early_start + p <= c:
                    self.job[i].task[j].start = early_start
                    self.job[i].task[j].end = early_start + p
                    self.decode_update_machine_idle(i, j, k, r, self.job[i].task[j].start)
                    break

    def constrain_limited_wait(self, i, index, mac=None):  # 等待时间有限约束
        for cursor, (j, j_next) in enumerate(zip(index[1:], index[:-1])):  # index为工序索引
            if self.direction == 0:  # 正向时间表
                limited_wait = self.job[i].task[j].limited_wait
            else:  # 反向时间表
                limited_wait = self.job[i].task[j_next].limited_wait
            cur_end = self.job[i].task[j].end
            next_start = self.job[i].task[j_next].start
            if next_start - cur_end - limited_wait > 0:  # 不满足等待时间有限约束
                if mac is None:
                    k = self.job[i].task[j].machine
                    p = self.job[i].task[j].duration
                else:
                    k = mac[i][j]
                    p = self.job[i].task[j].duration[self.job[i].task[j].machine.index(k)]
                delay_start = max([next_start - limited_wait - p, self.job[i].task[j].start])
                self.constrain_limited_wait_release_job(i, j, k, p)
                for r, (b, c) in enumerate(zip(self.machine[k].idle[0], self.machine[k].idle[1])):
                    early_start = max([delay_start, b])  # delay_start是满足等待时间有限约束的最早开始时间
                    if early_start + p <= c:
                        self.job[i].task[j].start = early_start
                        self.job[i].task[j].end = early_start + p
                        self.decode_update_machine_idle(i, j, k, r, self.job[i].task[j].start)
                        if next_start - self.job[i].task[j].end < 0:  # 相邻工序的时间间隔不合法
                            self.constrain_limited_wait_repair_interval(i, index, cursor, mac)
                            return False
                        break
        return True
