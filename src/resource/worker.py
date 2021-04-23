class Worker:  # 工人类
    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.start = None
        self.end = None

    def clear(self):  # 解码用：清除工人的任务信息
        self.start = None
        self.end = None
