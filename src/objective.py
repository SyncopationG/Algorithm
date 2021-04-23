class Objective:
    @staticmethod
    def makespan(scheduling):  # 工期
        return scheduling.makespan

    @staticmethod
    def tardiness(scheduling):  # 拖期
        a = 0
        for job in scheduling.job.values():
            a += max([0, job.end - job.due_date])
        return a
