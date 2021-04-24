from src import *


def makespan(schedule):
    return schedule.makespan


# low: 变量的下界; high: 变量的上界; dtype: 变量的数据类型; func: 目标函数
# max_or_min: 最大化或最小化(最大化: 值取0, 最小化: 值取1)
instance = 'kacem5'
func = makespan
n, m, p, tech, proc = Utils.string2data_fjsp(fjsp_benchmark.instance[instance])
length = sum(p)
low = [0] * length
high = [1] * length
dtype = [float] * length
my_problem = Utils.create_schedule(Fjsp, n, m, p, tech, proc, low=low, high=high, dtype=dtype)
my_problem.best_known = fjsp_benchmark.best_known[instance]
# # code = my_problem.code_operation_sequence(p)
# # # mac = my_problem.code_machine_assignment(n, p, tech)
# # mac = my_problem.code_machine_assignment_bml(n, m, p, tech, proc)
# info = my_problem.decode_operation_based_active(func, code, mac, direction=0)
# # info = my_problem.decode_only_operation_based_active(func, code, direction=0)
# print([machine.idle for machine in info.problem.machine.values()], "# machine idle")
# print(info.obj)
# info.ganttChart_png(lang=1, random_colors=False, with_start_end=False, with_operation=True)
# # De parameter
pop_size = 50
max_generation = 50
f, cr = 0.5, 0.5
max_or_min = 1
a = DeShopSchedule(pop_size, max_generation, f, cr, my_problem, func, max_or_min=max_or_min)
a.do_evolution()
a.best[0].ganttChart_png(filename=instance, lang=1, random_colors=False)
print([machine.idle for machine in a.best[0].problem.machine.values()], "# machine idle")
print(a.best[0].code, "# code")
print(a.best[0].mac, "# machine code")
#  这里的machine code是解码保存的
