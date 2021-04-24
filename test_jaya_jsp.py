from src import *


def makespan(schedule):
    return schedule.makespan


# low: 变量的下界; high: 变量的上界; dtype: 变量的数据类型; func: 目标函数
# max_or_min: 最大化或最小化(最大化: 值取0, 最小化: 值取1)
def do_jsp(instance):
    n, m, p, tech, proc = Utils.string2data_jsp(jsp_benchmark.instance[instance])
    # limited_wait = None
    # best_known = jsp_benchmark.best_known[instance]
    limited_wait = Utils.string2wait(Utils.load_text("./src/data/limited_wait_jsp/%s.txt" % instance), p)
    best_known = jsp_benchmark.best_known_limited_wait[instance]
    length = sum(p)
    func = makespan
    low = [0] * length
    high = [1] * length
    dtype = [float] * length
    max_or_min = 1
    my_problem = Utils.create_schedule(Jsp, n, m, p, tech, proc, limited_wait=limited_wait,
                                       best_known=best_known, low=low, high=high, dtype=dtype)
    # Jaya parameter
    pop_size = 50
    max_generation = 500
    a = JayaShopSchedule(pop_size, max_generation, my_problem, func, max_or_min=max_or_min)
    res, n_time = [], 10
    Utils.make_dir("./ResJaya")
    Utils.make_dir("./ResJaya/%s" % instance)
    for n_do in range(n_time):
        a.do_evolution()
        res.append([a.best[1], len(a.record[0]), a.best[0].problem.direction])
        Utils.save_code_to_txt("./ResJaya/%s/Code%s" % (instance, n_do), a.best[0].code)
        Utils.save_record_to_csv("./ResJaya/%s/Record%s" % (instance, n_do), a.record)
        print("Makespan: ", a.best[1])
    Utils.save_obj_to_csv("./ResJaya/%s.csv" % instance, res)


def main():
    for instance in INSTANCE_LIST.split():
        do_jsp(instance)


if __name__ == '__main__':
    main()
