import numpy as np


class Machine:  # 机器类
    def __init__(self, index, name=None):
        self.index = index
        self.name = name
        self.end = 0  # 机器上的最大完成时间
        self.idle = {0: [0, ], 1: [np.inf, ]}  # 机器空闲时间数据类型：dict，0：空闲开始时刻，1：空闲结束时刻
        self.index_list = []  # 解码用： 机器在编码中的位置索引（基于工序的编码）

    def clear(self):  # 解码用：清除机器的调度信息
        self.end = 0
        self.idle = {0: [0, ], 1: [np.inf, ]}
        self.index_list = []
