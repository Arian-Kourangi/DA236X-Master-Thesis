import numpy as np
from helpers.beziers import get_derivative_control_points_gurobi
from helpers.read_write_plan import csv_to_plan


SCENARIO = 'test1'
robot_names = ['snap', 'crackle']
object_name = 'pop'
original_plans = {'snap': {}, 'crackle': {}}
replans = {'snap': {}, 'crackle': {}}
for name in robot_names:
    rvars, hvars, ids, other_names = csv_to_plan(robot_name=name,scenario_name=SCENARIO,path=f'/home/arian/repos/thesis/impact_stl/impact_stl/planners/plans/{SCENARIO}')
    drvars = [get_derivative_control_points_gurobi(rvar) for rvar in rvars]
    dhvars = [get_derivative_control_points_gurobi(hvar) for hvar in hvars]
    original_plans[name] ={'rvars': rvars, 'hvars': hvars, 'drvars': drvars, 'dhvars': dhvars, 'ids': ids, 'other_names': other_names}
    print(f"Loaded plan for {name}:")
    replans[name] = np.load(f'/home/arian/repos/thesis/impact_stl/impact_stl/saved_logs/{SCENARIO}/{name}_replan_1.npz', allow_pickle=True)
    print(f"Loaded replan for {name}:")

print(original_plans['crackle']['hvars'][-1][0,-1])
print(replans['crackle']['hvars'][-1][0,-1])
print(original_plans['snap']['hvars'][-1][0,-1])
print(replans['snap']['hvars'][-1][0,-1])

data = np.load('/home/arian/repos/thesis/impact_stl/impact_stl/saved_logs/test1/crackle_1.npz', allow_pickle=True)

#print(data.files)
#t = data['t']
#x = data['x']
#xobj = data['xobj']
#inter = data['inter']
#print(t)
#print(inter)
#print(x.shape)