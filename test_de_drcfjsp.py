from src import *


def makespan(schedule):
    return schedule.makespan


# low: 变量的下界; high: 变量的上界; dtype: 变量的数据类型; func: 目标函数
# max_or_min: 最大化或最小化(最大化: 值取0, 最小化: 值取1)
instance = 'DMFJS01'
func = makespan
n, m, w, p, tech, worker, proc = Utils.string2data_drcfjsp(drcfjsp_benchmark.instance[instance])
length = sum(p)
low = [0] * length
high = [1] * length
dtype = [float] * length
my_problem = Utils.create_schedule(Fjsp, n, m, p, tech, proc, w, worker, low=low, high=high, dtype=dtype)
my_problem.best_known = drcfjsp_benchmark.best_known[instance]
# code = my_problem.code_operation_sequence(p)
# mac = my_problem.code_machine_assignment(n, p, tech)
# wok = my_problem.code_worker_assignment(n, p, tech, worker, mac)
# # mac = my_problem.code_machine_assignment_bml_worker(n, m, p, tech, proc)
# wok = my_problem.code_worker_assignment_bwl(n, w, p, tech, worker, proc, mac)
# info = my_problem.decode_operation_based_active(func, code, mac, wok, direction=0)
# print(code, "# code")
# print(mac, "# mac")
# print(wok, "# wok")
# 这里的code, mac, wok是生成的编码
# info = my_problem.decode_only_operation_based_active_worker(func, code, direction=0)
# print([machine.idle for machine in info.problem.machine.values()], "# machine idle")
# print(info.obj)
# print(info.mac, "# machine code")
# print(info.wok, "# worker code")
# 这里的machine code, worker code是解码保存的
# info.ganttChart_png(lang=1, random_colors=False, with_start_end=False, with_operation=True)
# De parameter
pop_size = 50
max_generation = 60
f, cr = 0.5, 0.5
max_or_min = 1
a = DeShopScheduleWorker(pop_size, max_generation, f, cr, my_problem, func, max_or_min=max_or_min)
a.do_evolution()
a.best[0].ganttChart_png(filename=instance, lang=1, random_colors=False, x_step=100)
print([machine.idle for machine in a.best[0].problem.machine.values()], "# machine idle")
print(a.best[0].code, "# code")
print(a.best[0].mac, "# machine code")
print(a.best[0].wok, "# worker code")
# 这里的machine code, worker code是解码保存的
