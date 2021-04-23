class Task:  # 加工任务（工序）类
    def __init__(self, index, machine, duration, name=None, limited_wait=None, worker=None):
        self.index = index
        self.machine = machine
        self.worker = worker
        self.duration = duration
        self.name = name
        self.limited_wait = limited_wait
        self.start = None  # 解码用：加工开始时间
        self.end = None  # 解码用：加工完成时间

    def clear(self):  # 解码用：清除工序的加工信息
        self.start = None
        self.end = None
