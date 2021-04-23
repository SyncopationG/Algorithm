from .task import Task


class Job:  # 工件类
    def __init__(self, index, due_date=None, name=None):
        self.index = index
        self.due_date = due_date
        self.name = name
        self.task = {}  # 工序集合
        self.nd = 0  # 解码用：已加工的工序数量
        self.index_list = []  # 解码用：工件在编码中的位置索引（基于工序的编码）

    def clear(self):  # 解码用：清除工件的加工信息
        for i in self.task.keys():
            self.task[i].clear()
        self.nd = 0
        self.index_list = [None for _ in range(self.nop)]

    @property
    def nop(self):  # 工序数量
        return len(self.task)

    def add_task(self, machine, duration, name=None, limited_wait=None, worker=None, index=None):  # 添加工序
        if index is None:
            index = self.nop
        self.task[index] = Task(index, machine, duration, name, limited_wait, worker)

    @property
    def start(self):  # 工件的加工开始时间
        return self.task[0].start
        # return min([task.start for task in self.task.values()])

    @property
    def end(self):  # 工件的加工完成时间
        return self.task[self.nop - 1].end
        # return max([task.end for task in self.task.values()])
