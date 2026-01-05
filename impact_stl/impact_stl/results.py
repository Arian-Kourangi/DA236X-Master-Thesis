import numpy as np
from helpers.beziers import get_derivative_control_points_gurobi
from helpers.read_write_plan import csv_to_plan
from robustness_calc import signed_distance_point_to_polygon
from helpers.beziers import eval_bezier
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

fm.fontManager.addfont("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf")
fm.fontManager.addfont("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf")
fm.fontManager.addfont("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-italic.otf")

mpl.rcParams.update({
    "font.size": 25,
    "axes.labelsize": 20,
    "axes.titlesize": 25,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes"],
    "mathtext.fontset": "stix",  # closest match for math
})
class Robot:
    def __init__(self, name, scenario, mpc= 'R', run_id = 1, verbose=False):
        self.name = name
        rvars, hvars, ids, other_names = csv_to_plan(robot_name=name,scenario_name=scenario,path=f'/home/arian/repos/thesis/impact_stl/impact_stl/planners/plans/{SCENARIO}')
        drvars = [get_derivative_control_points_gurobi(rvar) for rvar in rvars]
        dhvars = [get_derivative_control_points_gurobi(hvar) for hvar in hvars]
        self.original_plan ={'rvars': rvars, 'hvars': hvars, 'drvars': drvars, 'dhvars': dhvars, 'ids': ids, 'other_names': other_names}
        
        self.robot_start_pos = rvars[0][:2,0]
        rvars1,_,ids1,_ = csv_to_plan(robot_name='pop',scenario_name=scenario,path=f'/home/arian/repos/thesis/impact_stl/impact_stl/planners/plans/{SCENARIO}')
        self.original_plan_obj ={'rvars': rvars1, 'ids': ids1}
        
        self.object_start_pos = rvars1[0][:2,0]
        self.goal_pos = rvars1[-1][:2,-1]
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

SCENARIO = 'test4'
MPC = 'R'  # 'R' for reactive MPC, models the interaction and reacts to state of the object
            # N for nominal MPC, ignores the interaction altogether and just follows the plan
if SCENARIO not in ['test1','test2']: #These test do not have obstacles
    metrics = {'delta_V_pre':[], 'delta_V_post':[], 'delta_V_final':[], 'delta_pos_final':[], 'delta_T':[],'robustness':[]}
else: 
    metrics = {'delta_V_pre':[], 'delta_V_post':[], 'delta_V_final':[], 'delta_pos_final':[], 'delta_T':[]}

obstacles = {'test3': [np.array([0,10]), np.array([18,16])],
             'test4': [np.array([12,12]), np.array([18,18])]}

Verbose = False
for run_id in range(1,11):
    print(f"Analyzing run ID: {run_id}") if Verbose else None
    robots = {'snap':None, 'crackle':None}
    rhos = []
    for name in robots.keys():
        robot = Robot(name, SCENARIO,MPC, verbose=Verbose, run_id=run_id)
        robots[name] = robot

        ## Calculating metrics for each interaction
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
        
    # Robustness metrics
    if SCENARIO not in ['test1','test2']: #These test do not have obstacles
        lb = obstacles[SCENARIO][0]
        ub = obstacles[SCENARIO][1]
        for name, robot in robots.items():
            for log_idx in range(len(robot.log['t'])):
                pos = robot.log['x'][log_idx, 0:2]
                sd = signed_distance_point_to_polygon(pos, lb, ub)
                rhos.append(sd)
                obj_pos = robot.log['xobj'][log_idx, 0:2]
                sd_obj = signed_distance_point_to_polygon(obj_pos, lb, ub)
                rhos.append(sd_obj)
        
        robustness = np.min(rhos)
        metrics['robustness'].append(robustness)
        print(f"Robustness for run ID {run_id}: {robustness}") if Verbose else None

        

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

# Original robustness of the plan
rhos_plan = []

if SCENARIO not in ['test1','test2']: #These test do not have obstacles
    lb = obstacles[SCENARIO][0]
    ub = obstacles[SCENARIO][1]
    for name in robots.keys():
        robot = robots[name]
        for i in range(len(robot.original_plan['rvars'])):
            rvar = robot.original_plan['rvars'][i]
            evals = eval_bezier(rvar, N=100)
            for pos in evals.T:
                sd = signed_distance_point_to_polygon(pos[0:2], lb, ub)
                rhos_plan.append(sd)
        for i in range(len(robot.original_plan_obj['rvars'])):
            rvar = robot.original_plan_obj['rvars'][i]
            evals = eval_bezier(rvar, N=100)
            for pos in evals.T:
                sd = signed_distance_point_to_polygon(pos[0:2], lb, ub)
                rhos_plan.append(sd)
        
    robustness_plan = np.min(rhos_plan)
    print(f"Original plan robustness for scenario {SCENARIO}: {robustness_plan}")


#### Plotting #####
# Plot the setup first, including start pose of all robots and object, goal pose, and obstacles if ann
# make sure they are labeled in the plot, or include a legend

# Then plot the original planned trajectories of the the robots and objects
# Then the replanned curves
# Then the realised trajectories from the logs
# Enough to do it for one run_id per scenario
# Make seperate plots for Original, replanned, and realised trajectories
names = {'snap':r'$R_1$', 'crackle':r'$R_2$'}
if MPC == 'R':
    if SCENARIO == 'test1':
        offset = 5
        xlim = (-5,25)
        ylim = (-5,25)
    else:
        offset = 0.0
        xlim = (0,30)
        ylim = (0,30)
    fig, axes = plt.subplots(2, 2,  figsize=(14, 12))
    ax0, ax1, ax2, ax3 = axes.flatten()
    #fig, ax1 = plt.subplots(1, 1,  figsize=(7, 6))
    colors = {'snap':'black', 'crackle':'y', 'object':'red'}
    ls = {'snap':'k-', 'crackle':'y-', 'object':'r-'}
    inter_ls = {'snap':'cornflowerblue', 'crackle':'orange'}
    lw = 2
    thick_lw = 4
    ms = 15
    
    # Subplot 1: Original Plan
    # Plot obstacles
    if SCENARIO not in ['test1','test2']:
        lb = obstacles[SCENARIO][0]
        ub = obstacles[SCENARIO][1]
        rect = plt.Rectangle((lb[0], lb[1]), ub[0]-lb[0], ub[1]-lb[1], color='gray', alpha=0.8, label='Obstacle')
        ax1.add_patch(rect)
    # Plot original planned trajectories
    for name, robot in robots.items():
        evals = None
        for i,rvar in enumerate(robot.original_plan['rvars']):
            if robot.original_plan['ids'][i] != 'inter':
                evals = eval_bezier(rvar[0:2,:], N=100).T if evals is None else np.vstack((evals, eval_bezier(rvar[0:2], N=100).T))
        ax1.plot(evals[:,0] + offset, evals[:,1],ls[name], label=f'{names[name]} plan', linewidth=lw, zorder=1)
        labeled = False
        for i,rvar in enumerate(robot.original_plan['rvars']):
            if robot.original_plan['ids'][i] == 'inter':
                evals = eval_bezier(rvar[0:2,:], N=100).T #if evals is None else np.vstack((evals, eval_bezier(rvar[0:2], N=100).T))
                if not labeled:
                    ax1.plot(evals[:,0] + offset, evals[:,1],inter_ls[name], label=f'{names[name]} interaction', linewidth=thick_lw, zorder = 3)
                    labeled = True
                else:
                    ax1.plot(evals[:,0] + offset, evals[:,1],inter_ls[name], linewidth=thick_lw, zorder = 3)
        
        if name == ref_rob.name:
            evals = None
            for i,rvar in enumerate(robot.original_plan_obj['rvars']):
                if robot.original_plan_obj['ids'][i] != 'inter':
                    evals = eval_bezier(rvar, N=100).T if evals is None else np.vstack((evals, eval_bezier(rvar, N=100).T))
            ax1.plot(evals[:,0] + offset, evals[:,1], ls['object'], label=r'$O_1$ plan', linewidth=lw, zorder=2)
    # Plot start positions
    for name, robot in robots.items():
        ax1.plot(robot.robot_start_pos[0] + offset, robot.robot_start_pos[1], marker='o', color=colors[name], label=f'{names[name]} start', markersize=ms, zorder=4)
        if name == ref_rob.name:
            ax1.plot(robot.object_start_pos[0]+offset, robot.object_start_pos[1], marker='o', color=colors['object'], label=r'$O_1$ start', markersize=ms, zorder=4)
            ax1.plot(robot.goal_pos[0] + offset, robot.goal_pos[1], marker='*', color='green', label='Goal', markersize=ms, zorder=4)
    #ax1.set_title(f'Original Plans - Scenario {SCENARIO[-1]}')
    ax1.set_xlabel('X position [m]')
    ax1.set_ylabel('Y position [m]')
    ax1.set_xlim(xlim[0],xlim[1])
    ax1.set_ylim(ylim[0],ylim[1])
    ax1.set_xticks(np.arange(xlim[0],xlim[1]+1,5))
    ax1.set_yticks(np.arange(ylim[0],ylim[1]+1,5))
    ax1.set_xticklabels(np.arange(xlim[0],xlim[1]+1,5))
    ax1.set_yticklabels(np.arange(ylim[0],ylim[1]+1,5))
    ax1.legend(
    frameon=False,
    handlelength=1.5,
    labelspacing=0.3
)
    ax1.grid(True, alpha=0.3)
    #plt.tight_layout()
    #plt.show()
    #plt.savefig(f'/home/arian/repos/thesis/impact_stl/impact_stl/plots/{SCENARIO}_original_trajectories.pdf', bbox_inches="tight")
    # Subplot 2: Replanned Trajectories
    #fig, ax2 = plt.subplots(1, 1,  figsize=(7, 6))


    if SCENARIO not in ['test1','test2']:
        lb = obstacles[SCENARIO][0]
        ub = obstacles[SCENARIO][1]
        rect = plt.Rectangle((lb[0], lb[1]), ub[0]-lb[0], ub[1]-lb[1], color='gray', alpha=0.8, label='Obstacle')
        ax2.add_patch(rect)
    # Plot replanned trajectories
    for name, robot in robots.items():
        if SCENARIO not in ['test1','test2']:
            ids_printed = []
            evals = None
            labeled = False
            for i in range(len(robot.log['replans'])):
                if robot.log['replan_ids'][i] not in ids_printed:
                    pre = robot.log['replans'][i]
                    ids_printed.append(robot.log['replan_ids'][i])
                    evals = eval_bezier(pre[0:2], N=100).T #if evals is None else np.vstack((evals, eval_bezier(pre[0:2], N=100).T))
                    if not labeled:
                        ax2.plot(evals[:,0] + offset, evals[:,1],inter_ls[name], label=f'{names[name]} replans', linewidth=thick_lw, zorder = 3)
                        labeled = True
                    else:
                        ax2.plot(evals[:,0] + offset, evals[:,1],inter_ls[name], linewidth=thick_lw, zorder = 3)
        else:
            evals = None
            for rvar in robot.log['replans']:
                evals = eval_bezier(rvar[0:2,:], N=100).T if evals is None else np.vstack((evals, eval_bezier(rvar[0:2], N=100).T))
            ax2.plot(evals[:,0] + offset, evals[:,1],inter_ls[name], label=f'{names[name]} replans', linewidth=thick_lw, zorder = 3)
    # Plot start positions
    for name, robot in robots.items():
        ax2.plot(robot.robot_start_pos[0] + offset, robot.robot_start_pos[1], marker='o', color=colors[name], label=f'{names[name]} start', markersize=ms, zorder=4)
        if name == ref_rob.name:
            ax2.plot(robot.object_start_pos[0]+offset, robot.object_start_pos[1], marker='o', color=colors['object'], label=r'$O_1$ start', markersize=ms, zorder=4)
            ax2.plot(robot.goal_pos[0] + offset, robot.goal_pos[1], marker='*', color='green', label='Goal', markersize=ms, zorder=4)
    #ax2.set_title(f'Re-Plans - Scenario {SCENARIO[-1]}')
    ax2.set_xlabel('X position [m]')
    ax2.set_ylabel('Y position [m]')
    ax2.set_xlim(xlim[0],xlim[1])
    ax2.set_ylim(ylim[0],ylim[1])
    ax2.set_xticks(np.arange(xlim[0],xlim[1]+1,5))
    ax2.set_yticks(np.arange(ylim[0],ylim[1]+1,5))
    ax2.set_xticklabels(np.arange(xlim[0],xlim[1]+1,5))
    ax2.set_yticklabels(np.arange(ylim[0],ylim[1]+1,5))
    ax2.legend(
    frameon=False,
    handlelength=1.5,
    labelspacing=0.3)
    ax2.grid(True, alpha=0.3)
    #plt.tight_layout()
    #plt.show()
    #plt.savefig(f'/home/arian/repos/thesis/impact_stl/impact_stl/plots/{SCENARIO}_replanned_trajectories.pdf', bbox_inches="tight")




    #fig, ax3 = plt.subplots(1, 1,  figsize=(7, 6))

    # Subplot 3: Realized Trajectories
    # Plot obstacles
    if SCENARIO not in ['test1','test2']:
        lb = obstacles[SCENARIO][0]
        ub = obstacles[SCENARIO][1]
        rect = plt.Rectangle((lb[0], lb[1]), ub[0]-lb[0], ub[1]-lb[1], color='gray', alpha=0.8, label='Obstacle')
        ax3.add_patch(rect)
    # Plot realised trajectories from logs
    for name, robot in robots.items():
        ax3.plot(robot.log['x'][:,0] + offset, robot.log['x'][:,1],ls[name] , label=f'{names[name]} realized', linewidth=lw)
        if name == ref_rob.name:
            ax3.plot(robot.log['xobj'][:ref_rob.inter_sets[-1][-1],0] + offset, robot.log['xobj'][:ref_rob.inter_sets[-1][-1],1], ls['object'], label=r'$O_1$ realized', linewidth=lw, zorder=3)
    # Plot start positions
    for name, robot in robots.items():
        ax3.plot(robot.robot_start_pos[0] + offset, robot.robot_start_pos[1], marker='o', color=colors[name], label=f'{names[name]} start', markersize=ms, zorder =4)
        if name == ref_rob.name:
            ax3.plot(robot.object_start_pos[0]+offset, robot.object_start_pos[1], marker='o', color=colors['object'], label=r'$O_1$ start', markersize=ms, zorder=4)
            ax3.plot(robot.goal_pos[0] + offset, robot.goal_pos[1], marker='*', color='red', label='Goal', markersize=ms, zorder =4)
    #ax3.set_title(f'Realized Trajectories - Scenario {SCENARIO[-1]}')
    ax3.set_xlabel('X position [m]')
    ax3.set_ylabel('Y position [m]')
    ax3.set_xlim(xlim[0],xlim[1])
    ax3.set_ylim(ylim[0],ylim[1])
    ax3.set_xticks(np.arange(xlim[0],xlim[1]+1,5))
    ax3.set_yticks(np.arange(ylim[0],ylim[1]+1,5))
    ax3.set_xticklabels(np.arange(xlim[0],xlim[1]+1,5))
    ax3.set_yticklabels(np.arange(ylim[0],ylim[1]+1,5))
    ax3.legend(
    frameon=False,
    handlelength=1.5,
    labelspacing=0.3)
    ax3.grid(True, alpha=0.3)
    

 
    # Subplot 1: Original Plan
    # Plot obstacles
    if SCENARIO not in ['test1','test2']:
        lb = obstacles[SCENARIO][0]
        ub = obstacles[SCENARIO][1]
        rect = plt.Rectangle((lb[0], lb[1]), ub[0]-lb[0], ub[1]-lb[1], color='gray', alpha=0.8, label='Obstacle')
        ax0.add_patch(rect)
    # Plot start positions
    for name, robot in robots.items():
        ax0.plot(robot.robot_start_pos[0] + offset, robot.robot_start_pos[1], marker='o', color=colors[name], label=f'{names[name]} start', markersize=ms, zorder=4)
        if name == ref_rob.name:
            ax0.plot(robot.object_start_pos[0]+offset, robot.object_start_pos[1], marker='o', color=colors['object'], label=r'$O_1$ start', markersize=ms, zorder=4)
            ax0.plot(robot.goal_pos[0] + offset, robot.goal_pos[1], marker='*', color='green', label='Goal', markersize=ms, zorder=4)
    #ax0.set_title(f'Setup - Scenario {SCENARIO[-1]}')
    ax0.set_xlabel('X position [m]')
    ax0.set_ylabel('Y position [m]')
    ax0.set_xlim(xlim[0],xlim[1])
    ax0.set_ylim(ylim[0],ylim[1])
    ax0.set_xticks(np.arange(xlim[0],xlim[1]+1,5))
    ax0.set_yticks(np.arange(ylim[0],ylim[1]+1,5))
    ax0.set_xticklabels(np.arange(xlim[0],xlim[1]+1,5))
    ax0.set_yticklabels(np.arange(ylim[0],ylim[1]+1,5))
    ax0.legend(
    frameon=False,
    handlelength=1.5,
    labelspacing=0.3)
    ax0.grid(True, alpha=0.3)


    labels = ['(a)', '(b)', '(c)', '(d)']
    if SCENARIO == 'test3':
        for ax, lab in zip(axes.flatten(), labels):
            ax.text(
                0.98, 0.05,
                lab,
                transform=ax.transAxes,
                fontsize=25,
                fontweight="bold",
                va="bottom",
                ha="right"
            )
    else:
        for ax, lab in zip(axes.flatten(), labels):
            if SCENARIO == 'test2' and ax == ax1:
                ax.text(
                    0.98, 0.95,
                    lab,
                    transform=ax.transAxes,
                    fontsize=25,
                    fontweight="bold",
                    va="top",
                    ha="right"
                )
            else:
                ax.text(
                    0.02, 0.95,
                    lab,
                    transform=ax.transAxes,
                    fontsize=25,
                    fontweight="bold",
                    va="top",
                    ha="left"
                )
    
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'/home/arian/repos/thesis/impact_stl/impact_stl/plots/{SCENARIO}_trajectories.pdf', bbox_inches="tight")