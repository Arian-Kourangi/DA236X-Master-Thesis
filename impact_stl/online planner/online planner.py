import casadi as ca
from utilities.beziers import get_derivative_control_points_gurobi, eval_bezier, value_bezier, eval_t
from utilities.read_write_plan import csv_to_plan, plan_to_csv
# plotting imports
import scienceplots
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rc
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib import animation
from matplotlib.patches import Rectangle

import numpy as np
from scipy import optimize
import time
# Set up plot style
plt.style.use(['science'])
rc('text', usetex=True)
rc('font', family='times', size=35)

plt.rcParams['legend.frameon'] = True             # Enable legend frame
plt.rcParams['legend.facecolor'] = 'white'        # Set background color
plt.rcParams['legend.edgecolor'] = 'white'        # Set border color
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.loc'] = 'best'

# if true the constraints on velocity during the interaction are linear (on dr, dh is kept constant) otherwise they are on dq (nonlinear)
KEEP_VEL_CONSTRAINTS_LINEAR = False 

class TestOnlinePlanner():
    def __init__(self):
        #Size of world
        self.world_lb = np.array([0,0])
        self.world_ub = np.array([10,20])        
        self.robot_rvars,self.robot_hvars,self.robot_idvars,self.robot_other_names = csv_to_plan(robot_name='snap',
                                                                         scenario_name='minimal_test_throw_and_catch',
                                                                         path='/home/arian/repos/thesis/impact_stl/planner/plans')  

        self.obj_rvars,self.obj_hvars,self.obj_idvars,self.obj_other_names = csv_to_plan(robot_name='pop',
                                                                         scenario_name='minimal_test_throw_and_catch',
                                                                         path='/home/arian/repos/thesis/impact_stl/planner/plans')
        #Size = [nbzs][dim,ncp]

        #now we change the firs bezier of the object so it has an x velocity
        self.obj_rvars[0][0,0] = 0
        self.obj_rvars[0][0,1] = 2.5
        self.obj_rvars[1][0,0] = 2.5
        self.obj_rvars[1][0,1] = 5
        
        self.nbzs = len(self.robot_rvars)
        self.dim = 2
        self.ncp = self.robot_rvars[0].shape[1]
        self.world_tf = self.robot_hvars[-1][0,-1]
        #Remkaing to 2d for plotting
        self.obj_rvars = [self.obj_rvars[k][:2,:] for k in range(self.nbzs)]
        self.robot_rvars = [self.robot_rvars[k][:2,:] for k in range(self.nbzs)]
    
        self.robot_drvars = [get_derivative_control_points_gurobi(self.robot_rvars[k], 1) for k in range(self.nbzs)]
        self.obj_drvars = [get_derivative_control_points_gurobi(self.obj_rvars[k], 1) for k in range(self.nbzs)]
        self.robot_dhvars = [get_derivative_control_points_gurobi(self.robot_hvars[k], 1) for k in range(self.nbzs)]
        self.obj_dhvars = [get_derivative_control_points_gurobi(self.obj_hvars[k], 1) for k in range(self.nbzs)]
        self.obj_rad = 0.3
        self.rob_rad = 0.2
        
        self.replan()
        self.compute_trajectories()
        self.plot()
        self.animate()




    def replan(self):
        opti = ca.Opti()
        # lets pretend at the beginning of the pre curve we observe the new state of the object
        t_meas = 7.640449438202293
        #vel_meas = (self.obj_rvars[1][:,1] - self.obj_rvars[1][:,0]) / (self.obj_hvars[1][0,1] - self.obj_hvars[1][0,0])
        id, s = eval_t(self.obj_hvars, t_meas)
        #print("At time t = ", t_meas, " we are in segment ", id, " with local parameter s = ", s)
        pos_meas = value_bezier(self.obj_rvars[id], s)
        #print("At time t = ", t_meas, " the object is at position ", pos_meas)
        vel_meas = value_bezier(self.obj_drvars[id], s) / value_bezier(self.obj_dhvars[id], s)
        #print("At time t = ", t_meas, " the object has velocity ", vel_meas) 
        
        #Testing for sanity
        #vel_meas2 = (self.obj_rvars[1][:,1] - self.obj_rvars[1][:,0]) / (self.obj_hvars[1][0,1] - self.obj_hvars[1][0,0])
        #print("vel_meas2 = ", vel_meas2)
        #pos_meas2 = self.obj_rvars[1][:,0] + vel_meas2 * (t_meas - self.obj_hvars[1][0,0])
        #print("pos_meas2 = ", pos_meas2)
        
        predicted_pos = lambda t: pos_meas + vel_meas * (t - t_meas)
        
        #print("Predicted position at t = ", self.world_tf, " is ", predicted_pos(self.world_tf))
        n_cp = 6

        # New bezier variables for the robot, pre and inter curves
        rvars = [opti.variable(2,n_cp) for _ in range(2)] 
        hvars = [opti.variable(1,n_cp) for _ in range(2)]
        t_I = opti.variable() #time of interaction

        idvars = ['pre','inter']
        other_names = ['pop','pop']

        # dr and ddr
        drvars, ddrvars = [], []
        dhvars, ddhvars = [], []
        for idx in range(len(rvars)):
            drvars.append(get_derivative_control_points_gurobi(rvars[idx]))
            ddrvars.append(get_derivative_control_points_gurobi(rvars[idx],der_order=2))
            dhvars.append(get_derivative_control_points_gurobi(hvars[idx]))
            ddhvars.append(get_derivative_control_points_gurobi(hvars[idx],der_order=2))
        
        # increasing time
        for idx in range(len(dhvars)): 
            for i in range(dhvars[idx].shape[1]):
                opti.subject_to(dhvars[idx][0,i] >= 1e-1)

        #Contuinuity constraints
        opti.subject_to(rvars[0][:,-1] == rvars[-1][:,0])
        opti.subject_to(hvars[0][:,-1] == hvars[-1][:,0])
        opti.subject_to(drvars[0][:,-1] == drvars[-1][:,0])
        opti.subject_to(dhvars[0][:,-1] == dhvars[-1][:,0])


        #Initial conditions on robot (i don't have updated robot state, lets just assume it follows the preplanned one)
        #Get all pre idx
        pre_idxs = np.where(np.array(self.robot_idvars) == 'pre')[0]
        # Get the time of impact for all pre idxs, the impact time is the final time of the curve
        pre_tIs = [self.robot_hvars[pre_idx][0,-1] for pre_idx in pre_idxs]
        #Get the last pre idx before t_meas (or when the all is made whatever)
        pre_idx = next((pre_idxs[i] for i, tI in enumerate(pre_tIs) if tI > t_meas), len(pre_tIs)-1)

        #Beginning of pre curve ( not sure if this should be current time or planned beginning of pre curve??)
        
        t0 = self.robot_hvars[pre_idx][0,0]
        # Planned time of impact
        #tI = self.robot_hvars[pre_idx][0,-1]
        
        # Planned end of interaction
        tf = self.robot_hvars[pre_idx+1][0,-1]

        # Planned start and end positions and velocities
        x_start = self.robot_rvars[pre_idx][:,0]
        x_end = self.robot_rvars[pre_idx+1][:,-1]

        dr_start = self.robot_drvars[pre_idx][:,0] 
        dh_start = self.robot_dhvars[pre_idx][0,0]

        dr_end = self.robot_drvars[pre_idx+1][:,-1]
        dh_end = self.robot_dhvars[pre_idx+1][0,-1]

        #Initial position
        opti.subject_to(rvars[0][:,0] == x_start)
        opti.subject_to(hvars[0][0,0] == t0)
        #Initial velocity
        opti.subject_to(drvars[0][:,0] == dr_start)
        opti.subject_to(dhvars[0][0,0] == dh_start)
        #Time of interaction within bounds
        opti.subject_to(t_I >= t_meas)
        opti.subject_to(t_I <= tf)

        # End of pre curve should be the time of interaction
        opti.subject_to(hvars[0][0,-1] == t_I)
        opti.subject_to(hvars[1][0,0] == t_I)

        # Desired velocity after interaction - current measured velocity = desired change in velocity
        delta_V = dr_end/dh_end - vel_meas # Same as the desired object velocity at the end of the interaction - current object velocity
        travel_dir = np.arctan2(delta_V[1],delta_V[0])
        #print("Desired travel direction after interaction is ", travel_dir*180/np.pi, " degrees")
        unit_push_dir = np.array([np.cos(travel_dir), np.sin(travel_dir)])

        ###### Pre curve constraints

        # reach the predicted position at the end of pre curve and offset by the radii
        opti.subject_to(rvars[0][:,-1]== predicted_pos(t_I)-(self.rob_rad+self.obj_rad)*unit_push_dir) 
        
        # match its velocity
        opti.subject_to(drvars[0][:,-1] == vel_meas*dhvars[0][0,-1])


        ###### Interaction curve constraints
        #Final position ( again offset by the radii so the the objects is the one that needs to be at the target position not the robot)
        # Instead of enforcing exact equalities that may over-constrain the NLP,
        # create soft targets and penalize deviations in the objective.
        # Convert numpy targets to CasADi DM for safe mixing with CasADi variables.
        # keep in mind that x_end is taken from the robots planned trajectory, but it actually represents where the object should be.
        # Thats why we subtract the radii times unit_push_dir
        target_r_end = ca.DM(x_end - (self.rob_rad + self.obj_rad) * unit_push_dir)
        opti.subject_to(hvars[-1][0,-1] == tf)
        #target_h_tf = float(tf)
        target_dr_end = ca.DM(dr_end)
        target_dh_end = float(dh_end)
        # Note: penalties are added to the objective J later. Keep these as soft targets
        # so the solver can trade off exact satisfaction vs feasibility.

        y_pos = False # if false the direction of force in y demension is negative
        x_pos = False # if false the direction of force in x demension is negative

        # Push constraints - only push in the direction of desired velocity during the interaction
        if 0<= travel_dir <= np.pi:
            y_pos = True
        if -np.pi/2 <= travel_dir <= np.pi/2:
            x_pos = True
        #print("x_pos = ", x_pos, " y_pos = ", y_pos)

        # iterate over derivative control points (there are n_cp-1 derivative control points)
        # when comparing consecutive derivative control points we should stop one earlier
        # to avoid indexing past the last column
        for cp in range(drvars[1].shape[1]-1):
            if KEEP_VEL_CONSTRAINTS_LINEAR:
                if x_pos:
                    opti.subject_to(drvars[1][0,cp+1] - drvars[1][0,cp] >= 0) # increasing positive x velocity or decreasing negative x velocity
                else:
                    opti.subject_to(drvars[1][0,cp+1] - drvars[1][0,cp] <= 0) # decreasing positive x velocity or increasing negative x velocity
                if y_pos:
                    opti.subject_to(drvars[1][1,cp+1] - drvars[1][1,cp] >= 0) # increasing positive y velocity or decreasing negative y velocity
                else:
                    opti.subject_to(drvars[1][1,cp+1] - drvars[1][1,cp] <= 0) # decreasing positive y velocity or increasing negative y velocity

                opti.subject_to(dhvars[1][0,cp+1] == dhvars[1][0,cp]) # Constant time derivative for now

            else:
                if x_pos:
                    opti.subject_to(drvars[1][0,cp+1]/dhvars[1][0,cp+1]  - drvars[1][0,cp]/dhvars[1][0,cp]  >= 0) # increasing positive x velocity or decreasing negative x velocity
                else:
                    opti.subject_to(drvars[1][0,cp+1]/dhvars[1][0,cp+1]  - drvars[1][0,cp]/dhvars[1][0,cp]  <= 0) # decreasing positive x velocity or increasing negative x velocity
                if y_pos:
                    opti.subject_to(drvars[1][1,cp+1]/dhvars[1][0,cp+1]  - drvars[1][1,cp]/dhvars[1][0,cp]  >= 0) # increasing positive y velocity or decreasing negative y velocity
                else:
                    opti.subject_to(drvars[1][1,cp+1]/dhvars[1][0,cp+1]  - drvars[1][1,cp]/dhvars[1][0,cp]  <= 0) # decreasing positive y velocity or increasing negative y velocity


        # keep the robot in the world bounds
        for idx in range(len(rvars)):
            for i in range(rvars[idx].shape[1]):
                opti.subject_to(rvars[idx][0,i] >= self.world_lb[0])
                opti.subject_to(rvars[idx][0,i] <= self.world_ub[0])
                opti.subject_to(rvars[idx][1,i] >= self.world_lb[1])
                opti.subject_to(rvars[idx][1,i] <= self.world_ub[1])
        # Minimize the acceleration
        J = 0
        for idx in range(len(rvars)):
            for i in range(ddrvars[idx].shape[1]):
                J += ca.sumsqr(ddrvars[idx][:,i])
            for i in range(ddhvars[idx].shape[1]):
                J += ca.sumsqr(ddhvars[idx][0,i])
        # --- Soft penalties for final targets (relax exact equalities) ---
        # Weights (tune as needed)
        w_r = 1e3
        w_dr = 1e3
        w_h = 1e3
        w_dh = 1e3

        try:
            # target_* were prepared earlier (CasADi DM or floats)
            J += w_r * ca.sumsqr(rvars[-1][:,-1] - target_r_end)
            J += w_dr * ca.sumsqr(drvars[-1][:,-1] - target_dr_end)
            #J += w_h * (hvars[-1][0,-1] - target_h_tf)**2
            J += w_dh * (dhvars[-1][0,-1] - target_dh_end)**2
        except NameError:
            # If targets are not defined (shouldn't happen), skip adding penalties
            pass

        opti.minimize(J)
        # Set initial guesses
        for idx in range(len(rvars)):
            for k in range(rvars[idx].shape[0]):
                for i in range(rvars[idx].shape[1]):
                    opti.set_initial(rvars[idx][k,i], self.robot_rvars[pre_idx+idx][k,i])
                    opti.set_initial(hvars[idx][0,i], self.robot_hvars[pre_idx+idx][0,i])
            #for k in range(drvars[idx].shape[0]):
            #    for i in range(drvars[idx].shape[1]):
            #        opti.set_initial(drvars[idx][k,i], self.robot_drvars[pre_idx+idx][k,i])
            #        opti.set_initial(dhvars[idx][0,i], self.robot_dhvars[pre_idx+idx][0,i])

        #opti.set_initial(t_I, (t0 + tf)/2)


        #opts = {'ipopt.print_level': 0,
        #        'ipopt.tol': 1e-3,
        #        'ipopt.max_iter': 100,
        #        'print_time': 0, 'ipopt.sb': 'no'}
        #
        #opti.solver('ipopt',opts)
        qp_opts = {'osqp': {
            'max_iter': 100,
            'verbose': False,
            'eps_abs': 1e-3,
            'eps_rel': 1e-3,
            'adaptive_rho': False,
            'polish':True}, 'warm_start_primal': True, 'warm_start_dual': True
        }

        sqp_opts = {
            'max_iter': 12,
            'qpsol': 'osqp',
            'convexify_margin': 1e-4,
            'print_header': False,
            'print_time': False,
            'print_iteration': False,
            'qpsol_options': qp_opts
        }
        opti.solver('sqpmethod', sqp_opts)
        
        time_start = time.time()
        sol = opti.solve()
        time_end = time.time()
        print("Solved in ", time_end - time_start, " seconds")
        print("Optimal cost J = ", opti.value(J))
        print("Optimal time of interaction t_I = ", opti.value(t_I))
        #Extract the solution
        self.sol_robot_rvars = [sol.value(rvars[k]).reshape(2,6) for k in range(len(rvars))]
        self.sol_robot_hvars = [sol.value(hvars[k]).reshape(1,6) for k in range(len(hvars))]
        # Compare the replanned end velocity and postion with the desired ones
        print('Replanned end position = ', sol.value(rvars[-1][:,-1]), ' target position = ', x_end - (self.rob_rad + self.obj_rad) * unit_push_dir)
        print('Replanned end velocity = ', sol.value(drvars[-1][:,-1]/dhvars[-1][0,-1]), ' target velocity = ', dr_end/dh_end)

        #Adde the new curves to the existing ones
        self.robot_rvars[pre_idx] = self.sol_robot_rvars[0]
        self.robot_rvars[pre_idx+1] = self.sol_robot_rvars[1]
        self.robot_rvars[pre_idx+2][:,0] = self.sol_robot_rvars[1][:,-1] - (self.rob_rad + self.obj_rad) * unit_push_dir #updating the beginning of the next curve for plotting
        self.robot_drvars[pre_idx] = get_derivative_control_points_gurobi(self.robot_rvars[pre_idx], 1)
        self.robot_drvars[pre_idx+1] = get_derivative_control_points_gurobi(self.robot_rvars[pre_idx+1], 1)
        self.robot_hvars[pre_idx] = self.sol_robot_hvars[0]
        self.robot_hvars[pre_idx+1] = self.sol_robot_hvars[1]
        self.robot_dhvars[pre_idx] = get_derivative_control_points_gurobi(self.robot_hvars[pre_idx], 1)
        self.robot_dhvars[pre_idx+1] = get_derivative_control_points_gurobi(self.robot_hvars[pre_idx+1], 1)
        #Update the rest of the curves to match the new end velocity

        #For plotting
        self.obj_rvars[pre_idx+2][:,0] = self.sol_robot_rvars[1][:,-1] + (self.rob_rad + self.obj_rad) * unit_push_dir
        self.obj_hvars[pre_idx +2][0,0] = self.sol_robot_hvars[1][0,-1]
        self.obj_drvars[pre_idx+2][:,0] = self.sol_robot_rvars[1][:,-1]
        self.obj_dhvars[pre_idx+2][0,0] = self.sol_robot_hvars[1][0,-1]

    def compute_trajectories(self):
        N_eval = 100
        self.robot_rtraj = [eval_bezier(self.robot_rvars[k], N_eval) for k in range(self.nbzs)]
        self.obj_rtraj = [eval_bezier(self.obj_rvars[k], N_eval) for k in range(self.nbzs)]
        self.robot_htraj = [eval_bezier(self.robot_hvars[k], N_eval) for k in range(self.nbzs)]
        self.obj_htraj = [eval_bezier(self.obj_hvars[k], N_eval) for k in range(self.nbzs)]
        
        self.robot_drtraj = [eval_bezier(self.robot_drvars[k], N_eval) for k in range(self.nbzs)]
        self.obj_drtraj = [eval_bezier(self.obj_drvars[k], N_eval) for k in range(self.nbzs)]
        self.robot_dhtraj = [eval_bezier(self.robot_dhvars[k], N_eval) for k in range(self.nbzs)]
        self.obj_dhtraj = [eval_bezier(self.obj_dhvars[k], N_eval) for k in range(self.nbzs)]
        self.robot_dqtraj = [np.zeros((self.dim, N_eval)) for k in range(self.nbzs)]
        self.obj_dqtraj = [np.zeros((self.dim, N_eval)) for k in range(self.nbzs)]
        
        for k in range(self.nbzs):
            for i in range(self.dim):
                self.robot_dqtraj[k][i,:] = self.robot_drtraj[k][i,:]/self.robot_dhtraj[k][0,:]
                self.obj_dqtraj[k][i,:] = self.obj_drtraj[k][i,:]/self.obj_dhtraj[k][0,:]


    def plot(self):       
        
        # new layout for results figures
        fig = plt.figure(figsize=(20,10))
        gs = GridSpec(2,4, figure=fig)
        ax1 = fig.add_subplot(gs[0,0:2])
        ax2 = fig.add_subplot(gs[0,2:4])
        ax3 = fig.add_subplot(gs[1,0:2])
        ax4 = fig.add_subplot(gs[1,2:4])
        # ax5 = fig.add_subplot(gs[1,3])

        robot_ls = ['k-','k--','k:','k-.']
        object_ls = ['r-','r--','r:','r-.']
        lw = 4
        s = 15

        # X position vs time
        for bz in range(self.nbzs):
            #Only plot object if there is no interaction
            if 'inter' not in self.obj_idvars[bz]:
                ax1.plot(self.obj_htraj[bz][0,:],self.obj_rtraj[bz][0,:],object_ls[0],linewidth=lw)
                ax1.plot(self.obj_htraj[bz][0,0],self.obj_rtraj[bz][0,0],'ro',markersize=s)
        ax1.plot(self.obj_htraj[-1][0,-1],self.obj_rtraj[-1][0,-1],'ro',markersize=s)
        


        for bz in range(self.nbzs):
            #If there is an interaction, plot the robot trajectory in blue
            if 'inter' in self.robot_idvars[bz] and 'x' in self.robot_idvars[bz]:
                ax1.plot(self.robot_htraj[bz][0,:],self.robot_rtraj[bz][0,:],'b',linewidth=lw)
                ax1.plot(self.robot_htraj[bz][0,0],self.robot_rtraj[bz][0,0],'bo',markersize=s)
            else:
                ax1.plot(self.robot_htraj[bz][0,:],self.robot_rtraj[bz][0,:],robot_ls[0],linewidth=lw)
                ax1.plot(self.robot_htraj[bz][0,0],self.robot_rtraj[bz][0,0],'ko',markersize=s)
        ax1.plot(self.robot_htraj[-1][0,-1],self.robot_rtraj[-1][0,-1],'ko',markersize=s)
        # ax2.set_title("x-t plane")
        ax1.grid(True)
        ax1.set_xlabel(r"Time [s]")
        ax1.set_ylabel(r"x position [m]")

        # Y position vs time

        for bz in range(self.nbzs):
            #Only plot object if there is no interaction
            if 'inter' not in self.obj_idvars[bz]:    
                ax2.plot(self.obj_htraj[bz][0,:],self.obj_rtraj[bz][1,:],object_ls[0],linewidth=lw)
                ax2.plot(self.obj_htraj[bz][0,0],self.obj_rtraj[bz][1,0],'ro',markersize=s)
        ax2.plot(self.obj_htraj[-1][0,-1],self.obj_rtraj[-1][1,-1],'ro',markersize=s)
        
        for bz in range(self.nbzs):
            #If there is an interaction, plot the robot trajectory in blue
            if 'inter' in self.robot_idvars[bz] and 'y' in self.robot_idvars[bz] :
                ax2.plot(self.robot_htraj[bz][0,:],self.robot_rtraj[bz][1,:],'b',linewidth=lw)
                ax2.plot(self.robot_htraj[bz][0,0],self.robot_rtraj[bz][1,0],'bo',markersize=s)
            else:
                ax2.plot(self.robot_htraj[bz][0,:],self.robot_rtraj[bz][1,:],robot_ls[0],linewidth=lw)
                ax2.plot(self.robot_htraj[bz][0,0],self.robot_rtraj[bz][1,0],'ko',markersize=s)
        ax2.plot(self.robot_htraj[-1][0,-1],self.robot_rtraj[-1][1,-1],'ko',markersize=s)
        # ax3.set_title("y-t plane")
        ax2.grid(True)
        ax2.set_xlabel(r"Time [s]")
        ax2.set_ylabel(r"y position [m]")

        #X velocity vs time
        for bz in range(self.nbzs):
            #Only plot object if there is no interaction
            if 'inter' not in self.obj_idvars[bz]:
                ax3.plot(self.obj_htraj[bz][0,:],self.obj_dqtraj[bz][0,:],object_ls[0],linewidth=lw)
                ax3.plot(self.obj_htraj[bz][0,0],self.obj_dqtraj[bz][0,0],'ro',markersize=s)
        ax3.plot(self.obj_htraj[-1][0,-1],self.obj_dqtraj[-1][0,-1],'ro',markersize=s)
        
        for bz in range(self.nbzs):
            #If there is an interaction, plot the robot trajectory in blue
            if 'inter' in self.robot_idvars[bz] and 'x' in self.robot_idvars[bz]:
                ax3.plot(self.robot_htraj[bz][0,:],self.robot_dqtraj[bz][0,:],'b',linewidth=lw)
                ax3.plot(self.robot_htraj[bz][0,0],self.robot_dqtraj[bz][0,0],'bo',markersize=s)
            else:
                ax3.plot(self.robot_htraj[bz][0,:],self.robot_dqtraj[bz][0,:],robot_ls[0],linewidth=lw)
                ax3.plot(self.robot_htraj[bz][0,0],self.robot_dqtraj[bz][0,0],'ko',markersize=s)
        ax3.plot(self.robot_htraj[-1][0,-1],self.robot_dqtraj[-1][0,-1],'ko',markersize=s)
        # ax4.set_title("dx-t plane")
        ax3.grid(True)
        ax3.set_xlabel(r"Time [s]")
        ax3.set_ylabel(r"x velocity [m/s]")

        # Y velocity vs time
        for bz in range(self.nbzs):
            #Only plot object if there is no interaction
            if 'inter' not in self.obj_idvars[bz]:
                ax4.plot(self.obj_htraj[bz][0,:],self.obj_dqtraj[bz][1,:],object_ls[0],linewidth=lw)
                ax4.plot(self.obj_htraj[bz][0,0],self.obj_dqtraj[bz][1,0],'ro',markersize=s)
        ax4.plot(self.obj_htraj[-1][0,-1],self.obj_dqtraj[-1][1,-1],'ro',markersize=s)

        for bz in range(self.nbzs):
            #If there is an interaction, plot the robot trajectory in blue
            if 'inter' in self.robot_idvars[bz] and 'y' in self.robot_idvars[bz]:
                ax4.plot(self.robot_htraj[bz][0,:],self.robot_dqtraj[bz][1,:],'b',linewidth=lw)
                ax4.plot(self.robot_htraj[bz][0,0],self.robot_dqtraj[bz][1,0],'bo',markersize=s)
            else:
                ax4.plot(self.robot_htraj[bz][0,:],self.robot_dqtraj[bz][1,:],robot_ls[0],linewidth=lw)
                ax4.plot(self.robot_htraj[bz][0,0],self.robot_dqtraj[bz][1,0],'ko',markersize=s)
        ax4.plot(self.robot_htraj[-1][0,-1],self.robot_dqtraj[-1][1,-1],'ko',markersize=s)
        # ax4.set_title("dx-t plane")
        ax4.grid(True)
        ax4.set_xlabel(r"Time [s]")
        ax4.set_ylabel(r"y velocity [m/s]")

        fig.tight_layout()
        plt.savefig("/home/arian/repos/thesis/impact_stl/online planner/figures/plot.svg")

        #fig.show()

        # x-y plane
        fig_xy = plt.figure(figsize=(10, 10))
        axs = fig_xy.add_subplot(1, 1, 1)
        rect = Rectangle((self.world_lb[0],self.world_lb[1]),self.world_ub[0] - self.world_lb[0],self.world_ub[1] - self.world_lb[1],
                         linewidth=1,edgecolor='k',facecolor='none')
        axs.add_patch(rect)
        for bz in range(self.nbzs):
            axs.plot(self.robot_rtraj[bz][0, :], self.robot_rtraj[bz][1, :], robot_ls[0], linewidth=lw)
            axs.plot(self.robot_rtraj[bz][0, 0], self.robot_rtraj[bz][1, 0], 'ko', markersize=s)
            axs.plot(self.robot_rtraj[-1][0, -1],self.robot_rtraj[-1][1, -1], 'ko', markersize=s)
        for bz in range(self.nbzs):
            axs.plot(self.obj_rtraj[bz][0, :], self.obj_rtraj[bz][1, :], object_ls[0], linewidth=lw)
            axs.plot(self.obj_rtraj[bz][0, 0], self.obj_rtraj[bz][1, 0], 'ro', markersize=s)
            axs.plot(self.obj_rtraj[-1][0, -1], self.obj_rtraj[-1][1, -1], 'ro', markersize=s)
        axs.set_title("x-y plane")
        axs.set_xlabel("x [m]")
        axs.set_ylabel("y [m]")
        axs.grid(True)
        axs.set_aspect('equal', 'box')
        fig_xy.tight_layout()
        plt.savefig("/home/arian/repos/thesis/impact_stl/online planner/figures/xy_plane.svg")

        #fig_xy.show()
        #plt.pause(100)


    def evaluate_t(self,t):
            t_array_robots = np.array([self.robot_hvars[bzr][0,0] for bzr in range(self.nbzs)])
            t_array_objects = np.array([self.obj_hvars[bzo][0,0] for bzo in range(self.nbzs)]) 

            idxs_robots = np.where(t_array_robots<=t)[0][-1]
            idxs_objects = np.where(t_array_objects<=t)[0][-1]

            errors_robot= lambda s: value_bezier(self.robot_htraj[idxs_robots],s)[0] - t
            errors_object =lambda s: value_bezier(self.obj_htraj[idxs_objects],s)[0] - t

            s_robot = optimize.root_scalar(errors_robot,bracket=[0,1]).root
            s_object = optimize.root_scalar(errors_object,bracket=[0,1]).root

            return idxs_robots, s_robot, idxs_objects, s_object
    
    def animate(self):
        Neval = 250
        self.t_range = np.linspace(0,self.world_tf,Neval)

        self.fig_anim = plt.figure(figsize=(10,10))
        self.ax_anim = plt.axes()
        anim = animation.FuncAnimation(self.fig_anim,self._animate_update,
                                        frames=Neval,interval=self.world_tf/Neval*1e6*2)
        anim.save("/home/arian/repos/thesis/impact_stl/online planner/figures/animation.mp4",writer="ffmpeg", fps=Neval/self.world_tf)


    def _animate_update(self,i):
        t = self.t_range[i]
        idxs_robot, s_robot, idxs_object, s_object = self.evaluate_t(t)

        self.ax_anim.clear()
        rect = Rectangle((self.world_lb[0],self.world_lb[1]),self.world_ub[0] - self.world_lb[0],self.world_ub[1] - self.world_lb[1],
                         linewidth=1,edgecolor='k',facecolor='none')
        self.ax_anim.add_patch(rect)
        # plot circle
        c = (value_bezier(self.robot_rtraj[idxs_robot],s_robot)[0],
            value_bezier(self.robot_rtraj[idxs_robot],s_robot)[1])
        circle = plt.Circle(c,self.rob_rad,fill=False,color='k')
        self.ax_anim.add_patch(circle)
        self.ax_anim.text(
        c[0], c[1] + 0.25,  # slightly above the circle
        'RobTheRobot',
        color='k', fontsize=12, ha='center', va='bottom', weight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        # plot trajectory 
        for bz in range(self.nbzs):
            #Only plot the traj if interaction
            if 'inter' in self.robot_idvars[bz]:
                self.ax_anim.plot(self.robot_rtraj[bz][0,:],self.robot_rtraj[bz][1,:],'b', linewidth=2)

        # plot circle
        c = (value_bezier(self.obj_rtraj[idxs_object],s_object)[0],
            value_bezier(self.obj_rtraj[idxs_object],s_object)[1])
        circle = plt.Circle(c,self.obj_rad,fill=False,color='r')
        self.ax_anim.add_patch(circle)
        # plot trajectory
        for bz in range(self.nbzs):
            #Only plot object if there is no interaction
            if 'inter' not in self.obj_idvars[bz]:
                self.ax_anim.plot(self.obj_rtraj[bz][0,:],self.obj_rtraj[bz][1,:],'r',linewidth=2)
        self.ax_anim.set_title("x-y plane")
        self.ax_anim.set_xlabel("x [m]")
        self.ax_anim.set_ylabel("y [m]")
        self.ax_anim.set_aspect("equal")
test = TestOnlinePlanner()
