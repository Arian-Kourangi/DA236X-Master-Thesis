import numpy as np
from helpers.beziers import get_derivative_control_points_gurobi
from helpers.read_write_plan import csv_to_plan


class Robot:
    def __init__(self, name, scenario, mpc= 'R', run_id = 1, verbose=False):
        self.name = name
        rvars, hvars, ids, other_names = csv_to_plan(robot_name=name,scenario_name=scenario,path=f'/home/arian/repos/thesis/impact_stl/impact_stl/planners/plans/{SCENARIO}')
        drvars = [get_derivative_control_points_gurobi(rvar) for rvar in rvars]
        dhvars = [get_derivative_control_points_gurobi(hvar) for hvar in hvars]
        self.original_plan ={'rvars': rvars, 'hvars': hvars, 'drvars': drvars, 'dhvars': dhvars, 'ids': ids, 'other_names': other_names}
        print(f"Loaded plan for {name}:") if verbose else None
        self.replan = np.load(f'/home/arian/repos/thesis/impact_stl/impact_stl/saved_logs/{scenario}/{mpc}/{name}_replan_{run_id}.npz', allow_pickle=True)
        print(f"Loaded replan for {name}:") if verbose else None

        self.log = np.load(f'/home/arian/repos/thesis/impact_stl/impact_stl/saved_logs/{scenario}/{mpc}/{name}_{run_id}.npz', allow_pickle=True)

        self.inter_idxs = np.where(np.array(self.original_plan['ids']) == 'inter')[0]
        print(f"Interaction indices for {name}: {self.inter_idxs}") if verbose else None

        inter_log_idxs = np.where(np.array(self.log['inter']) == 1)[0]
        print(inter_log_idxs) if verbose else None

        start = None
        end = None
        inter_sets = []
        for idx in inter_log_idxs:
            if start is None:
                start = idx
            end = idx
            if (idx + 1) not in inter_log_idxs:
                inter_sets.append((start, end))
                start = None
                end = None
        self.inter_sets = inter_sets
        print(f"Found {inter_sets} interaction sets for {name}.") if verbose else None
        self.T_final_plan = self.original_plan['hvars'][self.inter_idxs[-1]][0, -1]
        print(f"Final time for {name}: {self.T_final_plan}") if verbose else None


SCENARIO = 'test1'
MPC = 'R'  # 'R' for reactive MPC, models the interaction and reacts to state of the object
            # N for nominal MPC, ignores the interaction altogether and just follows the plan
metrics = {'delta_V_pre':[], 'delta_V_post':[], 'delta_V_final':[], 'delta_pos_final':[], 'delta_T':[]}
Verbose = False
for run_id in range(1,11):
    print(f"Analyzing run ID: {run_id}") if Verbose else None
    robots = {'snap':None, 'crackle':None}
    for name in robots.keys():
        robot = Robot(name, SCENARIO,MPC, verbose=Verbose, run_id=run_id)
        robots[name] = robot

        for i in range(len(robot.inter_sets)):
            start_idx, end_idx = robot.inter_sets[i]

            #Comiparing delta_V_pre of robot and object
            robot_vel = robot.log['x'][start_idx, 3:6]
            object_vel = robot.log['xobj'][start_idx, 3:6]
            print(f"Robot {robot.name} interaction {i} pre-impact velocities: robot {robot_vel}, object {object_vel}") if Verbose else None
            delta_V_pre = np.linalg.norm(robot_vel - object_vel)
            metrics['delta_V_pre'].append(delta_V_pre)
            print(f"Robot {robot.name} interaction {i} delta_V_pre: {delta_V_pre}") if Verbose else None

            # Comparing delta_V_post of object and the desired post-interaction velocity from the original plan
            object_vel_post = robot.log['xobj'][end_idx, 3:6]
            desired_vel_post = robot.original_plan['drvars'][robot.inter_idxs[i]][:, -1]/robot.original_plan['dhvars'][robot.inter_idxs[i]][0, -1]
            print(f"Robot {robot.name} interaction {i} post-impact velocities: object {object_vel_post}, desired {desired_vel_post}") if Verbose else None
            delta_V_post = np.linalg.norm(object_vel_post - desired_vel_post)
            metrics['delta_V_post'].append(delta_V_post)
            print(f"Robot {robot.name} interaction {i} delta_V_post: {delta_V_post}") if Verbose else None


    # Find the robot with the largest final time, compare the planned vel, pos and time
    max_T_final = -np.inf
    for name, robot in robots.items():
        if robot.T_final_plan > max_T_final:
            max_T_final = robot.T_final_plan
            ref_rob = robot

    #Final velocity and position of the object

    final_obj_vel_plan = ref_rob.original_plan['drvars'][ref_rob.inter_idxs[-1]][:, -1]/ref_rob.original_plan['dhvars'][ref_rob.inter_idxs[-1]][0, -1]
    final_obj_pos_plan = ref_rob.original_plan['rvars'][ref_rob.inter_idxs[-1]][:, -1]
    final_time_plan = ref_rob.T_final_plan

    print(f"Reference robot {ref_rob.name} final planned object velocity: {final_obj_vel_plan}") if Verbose else None
    print(f"Reference robot {ref_rob.name} final planned object position: {final_obj_pos_plan}") if Verbose else None
    print(f"Reference robot {ref_rob.name} final planned time: {final_time_plan}") if Verbose else None

    start,end = ref_rob.inter_sets[-1]
    final_obj_vel_log = ref_rob.log['xobj'][end, 3:6]
    final_obj_pos_log = ref_rob.log['xobj'][end, 0:3]
    final_time_log = ref_rob.log['t'][end]
    print(f"Reference robot {ref_rob.name} final logged object velocity: {final_obj_vel_log}") if Verbose else None
    print(f"Reference robot {ref_rob.name} final logged object position: {final_obj_pos_log}") if Verbose else None
    print(f"Reference robot {ref_rob.name} final logged time: {final_time_log}") if Verbose else None

    delta_V_final = np.linalg.norm(final_obj_vel_log - final_obj_vel_plan)
    metrics['delta_V_final'].append(delta_V_final)
    delta_pos_final = np.linalg.norm(final_obj_pos_log - final_obj_pos_plan)
    metrics['delta_pos_final'].append(delta_pos_final)
    delta_T = np.abs(final_time_log - final_time_plan)
    metrics['delta_T'].append(delta_T)
    print(f"Final delta_V: {delta_V_final}") if Verbose else None
    print(f"Final delta_pos: {delta_pos_final}") if Verbose else None
    print(f"Final delta_T: {delta_T}") if Verbose else None

print("=======================================")
print(f'Metrics for scenario {SCENARIO}, MPC type {MPC}:')
# Printing out all the metrics, their means and medians
for key, values in metrics.items():
    mean_val = np.mean(values)
    median_val = np.median(values)
    #print(f"Values for metric {key}: {values}") if Verbose else None
    print(f"Metric {key}:  mean = {mean_val}, median = {median_val}")



#print(data.files)
#t = data['t']
#x = data['x']
#xobj = data['xobj']
#inter = data['inter']
#print(t)
#print(inter)
#print(x.shape)