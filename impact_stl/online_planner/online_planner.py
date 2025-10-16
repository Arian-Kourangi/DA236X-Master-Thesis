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
KEEP_VEL_CONSTRAINTS_LINEAR = True 

class TestOnlinePlanner():
    def __init__(self):
        #Size of world
        self.world_lb = np.array([0,0])
        self.world_ub = np.array([10,20])
        self.dq_lb = np.array([-2,-2])
        self.dq_ub = np.array([2,2])        
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
        self.obj_rad = 0.2
        self.rob_rad = 0.2
        
        self.replan()
        self.compute_trajectories()
        self.plot()
        self.animate()
        #self.save_plan()

    def get_pre_Idxs(self, t_meas):
        """
        Args:
            t_meas (float): time of measurement
        Returns:
            rob_pre_idx (int): index of the first pre curve after t_meas
            obj_pre_idx (int): index of the next pre curve for the object after t_meas
            obj_next_pre_idx (int): index of the next pre curve for the object after obj_pre_idx
        """
        #Initial conditions on robot (i don't have updated robot state, lets just assume it follows the preplanned one)
        
        #Get all pre idx
        rob_pre_idxs = np.where(np.array(self.robot_idvars) == 'pre')[0]
        # Get the time of impact for all pre idxs, the impact time is the final time of the curve
        rob_pre_tIs = [self.robot_hvars[pre_idx][0,-1] for pre_idx in rob_pre_idxs]
        #Get the last pre idx before t_meas (or when the all is made whatever)
        rob_pre_idx = next((rob_pre_idxs[i] for i, tI in enumerate(rob_pre_tIs) if tI > t_meas), len(rob_pre_tIs)-1)

        # Get obj pre index
        obj_pre_idxs = np.where(np.array(self.obj_idvars) == 'pre')[0]
        obj_pre_TIs = [self.obj_hvars[pre_idx][0,-1] for pre_idx in obj_pre_idxs]
        obj_pre_future_idx = [obj_pre_idxs[i] for i, tI in enumerate(obj_pre_TIs) if tI > t_meas]
        obj_pre_idx = obj_pre_future_idx[0] # if there are multiple future pre curves take the first one
        obj_next_pre_idx = obj_pre_future_idx[1] if len(obj_pre_future_idx) > 1 else obj_pre_future_idx[0]
        return rob_pre_idx, obj_pre_idx, obj_next_pre_idx


    def replan(self):
        opti = ca.Opti()

        ### This part is instead of measurement form actual sensors
        # lets pretend at the beginning of the pre curve we observe the new state of the object
        t_meas = 7.640449438202293
        id, s = eval_t(self.obj_hvars, t_meas)
        pos_meas = value_bezier(self.obj_rvars[id], s)
        vel_meas = value_bezier(self.obj_drvars[id], s) / value_bezier(self.obj_dhvars[id], s)
        #######

        # Function to predict the position of the object at time t based on current measurement and constant velocity model
        predicted_pos = lambda t: pos_meas + vel_meas * (t - t_meas)
        n_cp = 6

        # Used for measuring the time of the entire replanning
        Beginning_time = time.time()

        # New bezier variables for the robot, pre and inter curves
        rvars = [opti.variable(2,n_cp) for _ in range(2)] 
        hvars = [opti.variable(1,n_cp) for _ in range(2)]
        t_I = opti.variable() #time of interaction

        # Construct the derivative control points
        drvars, ddrvars = [], []
        dhvars, ddhvars = [], []
        for idx in range(len(rvars)):
            drvars.append(get_derivative_control_points_gurobi(rvars[idx]))
            ddrvars.append(get_derivative_control_points_gurobi(rvars[idx],der_order=2))
            dhvars.append(get_derivative_control_points_gurobi(hvars[idx]))
            ddhvars.append(get_derivative_control_points_gurobi(hvars[idx],der_order=2))
        
        # increasing time
        for i in range(dhvars[0].shape[1]):
            opti.subject_to(dhvars[0][0,i] >= 1*1e-1)
        ## Tigher constraing for interaction curve to minimize acceleration
        for i in range(dhvars[1].shape[1]):
            opti.subject_to(dhvars[1][0,i] >= 30*1e-1)

        #Contuinuity constraints between pre and interaction curve
        opti.subject_to(rvars[0][:,-1] == rvars[-1][:,0])
        opti.subject_to(hvars[0][:,-1] == hvars[-1][:,0])
        opti.subject_to(drvars[0][:,-1] == drvars[-1][:,0])
        opti.subject_to(dhvars[0][:,-1] == dhvars[-1][:,0])
        
        # keep the robot in the world bounds
        for idx in range(len(rvars)):
            for i in range(rvars[idx].shape[1]):
                opti.subject_to(rvars[idx][0,i] >= self.world_lb[0])
                opti.subject_to(rvars[idx][0,i] <= self.world_ub[0])
                opti.subject_to(rvars[idx][1,i] >= self.world_lb[1])
                opti.subject_to(rvars[idx][1,i] <= self.world_ub[1])
        
        #Velocity constraints - keep the velocity within bounds for the pre curve
        for i in range(drvars[0].shape[1]):
            opti.subject_to(drvars[0][0,i] <= self.dq_ub[0]*dhvars[0][0,i])
            opti.subject_to(drvars[0][0,i] >= self.dq_lb[0]*dhvars[0][0,i])
            opti.subject_to(drvars[0][1,i] <= self.dq_ub[1]*dhvars[0][0,i])
            opti.subject_to(drvars[0][1,i] >= self.dq_lb[1]*dhvars[0][0,i])

        #Velocity constraints - keep the velocity within bounds for the interaction curve, 
        # now we want the max velocity to half of actual max, since the robot is also pushing the object
        for i in range(drvars[1].shape[1]):
            opti.subject_to(drvars[1][0,i] <= 0.5*self.dq_ub[0]*dhvars[1][0,i])
            opti.subject_to(drvars[1][0,i] >= 0.5*self.dq_lb[0]*dhvars[1][0,i])
            opti.subject_to(drvars[1][1,i] <= 0.5*self.dq_ub[1]*dhvars[1][0,i])
            opti.subject_to(drvars[1][1,i] >= 0.5*self.dq_lb[1]*dhvars[1][0,i])
        
        # Get the idx of the robots next pre curve after t_meas, and the two next pre curves of the object
        pre_idx, obj_pre_idx, obj_next_pre = self.get_pre_Idxs(t_meas)

        ###### FROM PLANNED TRAJECTORY ######
        # Planned start time of pre curve        
        t0 = self.robot_hvars[pre_idx][0,0]
        # Planned time of impact
        tI = self.robot_hvars[pre_idx][0,-1]
        # Planned end of interaction
        tf = self.robot_hvars[pre_idx+1][0,-1]

        # Planned start and end positions and velocities
        x_start = self.robot_rvars[pre_idx][:,0]
        # This is the end position of the object at the end of the interaction curve, but because the offline planner assumes point masses the robots is the same
        x_end = self.robot_rvars[pre_idx+1][:,-1]

        dr_start = self.robot_drvars[pre_idx][:,0] 
        dh_start = self.robot_dhvars[pre_idx][0,0]

        dr_end = self.robot_drvars[pre_idx+1][:,-1]
        dh_end = self.robot_dhvars[pre_idx+1][0,-1]
        #####################################
        
        #Initial position and time of the robot pre curve
        opti.subject_to(rvars[0][:,0] == x_start)
        opti.subject_to(hvars[0][0,0] == t0)
        #Initial velocity of the robot pre curve
        opti.subject_to(drvars[0][:,0] == dr_start)
        opti.subject_to(dhvars[0][0,0] == dh_start)
        #Time of interaction within bounds
        opti.subject_to(t_I >= t_meas)
        #opti.subject_to(t_I <= tf)

        # End of pre curve and beginning of inter curve should be the time of interaction
        opti.subject_to(hvars[0][0,-1] == t_I)
        opti.subject_to(hvars[1][0,0] == t_I)

        # Desired velocity after interaction - current measured velocity = desired change in velocity
        delta_V = dr_end/dh_end - vel_meas # Same as the desired object velocity at the end of the interaction - current object velocity
        # Get the interaction angle
        travel_dir = np.arctan2(delta_V[1],delta_V[0])
        # Unit vector in the direction of travel ( to get the ratio of the change in velocity in the two dimensions)
        unit_push_dir = np.array([np.cos(travel_dir), np.sin(travel_dir)])

        ###### Pre curve constraints

        # reach the predicted position at the end of pre curve and offset by the radii
        opti.subject_to(rvars[0][:,-1]== predicted_pos(t_I)-(self.rob_rad+self.obj_rad)*unit_push_dir) 
        
        # match its velocity
        opti.subject_to(drvars[0][:,-1] == vel_meas*dhvars[0][0,-1])



        ###### Interaction curve constraints
        # Final position ( again offset by the radii so the the objects is the one that needs to be at the target position not the robot)
        # Instead of enforcing exact equalities that may over-constrain the NLP,
        # create soft targets and penalize deviations in the objective.
        # Convert numpy targets to CasADi DM for safe mixing with CasADi variables.
        # keep in mind that x_end is taken from the robots planned trajectory, but it actually represents where the object should be.
        # Thats why we subtract the radii times unit_push_dir
        
        ## Also add that the vectors are parallel
        next_obj_int_pos = self.obj_rvars[obj_next_pre][:,-1]
        #Vector between planned end of interaction and the next object interaction position
        v = next_obj_int_pos - x_end
        # The position of the robot at the end of the interaction plus the radii*unit_push is where the object is at the end of this interaction
        object_end = rvars[1][:,-1] + (self.rob_rad + self.obj_rad) * unit_push_dir
        #Create a vector between the replanned end of interaction and the next object interaction position
        new_v = next_obj_int_pos - object_end
        #Make sure these are parallel
        opti.subject_to(v[0]*new_v[1] - v[1]*new_v[0] == 0) # cross product = 0 means they are colinear/parallel


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
        
        # Make sure the ratio of change in y velocity to change in x velocity is the same as the desired delta_V
        # This ensures that the robot pushes in the direction of desired velocity i.e the interaction angle is constant
        for i in range(drvars[1].shape[1]-1):
            d_x = drvars[1][0,cp+1] - drvars[1][0,cp]
            d_y = drvars[1][1,cp+1] - drvars[1][1,cp]
            opti.subject_to(d_x * delta_V[1] - d_y * delta_V[0] == 0) # cross product = 0 means they are colinear
        
        # Minimize the acceleration
        J = 0
        # For pre curve we can have smaller weights
        for i in range(ddrvars[0].shape[1]):
            J += ca.sumsqr(ddrvars[0][:,i])
        for i in range(ddhvars[0].shape[1]):
            J += ca.sumsqr(ddhvars[0][0,i])
        
        # For interaction curve we want to minimize acceleration more
        w_acc = 1e2
        for i in range(ddrvars[1].shape[1]):
            J += w_acc*ca.sumsqr(ddrvars[1][:,i])
        for i in range(ddhvars[1].shape[1]):
            J += w_acc*ca.sumsqr(ddhvars[1][0,i])
        
        # --- Soft penalties for final targets (relax exact equalities) ---
        # Weights (tune as needed)
        w_r = 1e5
        w_dr = 1e3
        w_h = 1e-1
        w_dh = 1e3
        # We also want the end of the itneraction to be as close as possible to the planned end of interaction
        target_r_end = ca.DM(x_end)
        target_h_tf = float(tf)
        target_dr_end = ca.DM(dr_end)
        target_dh_end = float(dh_end)
        try:
            # target_* were prepared earlier (CasADi DM or floats)
            object_pos = rvars[1][:,-1]+(self.rob_rad + self.obj_rad) * unit_push_dir
            J += w_r * ca.sumsqr(object_pos- target_r_end)
            J += w_dr * ca.sumsqr(drvars[-1][:,-1] - target_dr_end)
            J += w_h * (hvars[-1][0,-1] - target_h_tf)**2
            J += w_h * (t_I - tI )**2  # also penalize deviation from t_I at start of interaction curve
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

        qp_opts = {'osqp': {
            'max_iter': 1000,
            'verbose': False,
            'eps_abs': 1e-3,
            'eps_rel': 1e-3,
            'adaptive_rho': True,
            'polish':True}, 'warm_start_primal': True, 'warm_start_dual': True
        }

        sqp_opts = {
            'max_iter': 100,
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
        print('Planned end time = ', self.robot_hvars[pre_idx+1][0,-1], ' replanned end time = ', sol.value(hvars[1][0,-1]))
        #Extract the solution
        self.sol_robot_rvars = [sol.value(rvars[k]).reshape(2,n_cp) for k in range(len(rvars))]
        self.sol_robot_hvars = [sol.value(hvars[k]).reshape(1,n_cp) for k in range(len(hvars))]
        # Compare the replanned end velocity and postion with the desired ones
        print('Replanned end position = ', sol.value(rvars[-1][:,-1]), ' target position = ', x_end - (self.rob_rad + self.obj_rad) * unit_push_dir)
        print('Replanned end velocity = ', sol.value(drvars[-1][:,-1]/dhvars[-1][0,-1]), ' target velocity = ', dr_end/dh_end)
        self.update_plan(pre_idx, obj_pre_idx, unit_push_dir)
        Stop_time = time.time()
        print("Time to set up and solve the replanning problem: ", Stop_time - Beginning_time, " seconds")

    def update_plan(self, pre_idx, obj_pre_idx, unit_push_dir):

        """
        Args:
            pre_idx (int): index of the first pre curve after t_meas
            obj_pre_idx (int): index of the next pre curve for the object after t_meas
            unit_push_dir (np.array): unit vector in the direction of interaction angle
        Returns:
            Updates the robot and object bezier curves with the replanned ones
        Note: this function assumes that the replanning is done only for one robot and one object
        """
        #Added the new curves to the existing ones
        self.robot_rvars[pre_idx] = self.sol_robot_rvars[0]
        self.robot_rvars[pre_idx+1] = self.sol_robot_rvars[1]
        self.robot_rvars[pre_idx+2][:,0] = self.sol_robot_rvars[1][:,-1] # Contunuity

        # If I allow the time at the end of the interaction to shift, I need to propogate this change to the rest of the curves
        # Compute the time difference at the end of the interaction curve
        self.end_time_diff  = self.sol_robot_hvars[1][0,-1] - self.robot_hvars[pre_idx+1][0,-1]
        ## Propogate time change to the rest of the curves ( this should be done to all robots probably)
        for k in range(pre_idx+2, self.nbzs):
            self.robot_hvars[k][0,:] += self.end_time_diff
        self.robot_hvars[pre_idx] = self.sol_robot_hvars[0]
        self.robot_hvars[pre_idx+1] = self.sol_robot_hvars[1]
        self.robot_hvars[pre_idx+2][0,0] = self.sol_robot_hvars[1][0,-1] # Contunuity

        # Propogate the time change to the rest of the object curves
        for k in range(obj_pre_idx+2, self.nbzs):
            self.obj_hvars[k][0,:] += self.end_time_diff

        #Recompute all derivative control points for the robot
        self.robot_drvars = [get_derivative_control_points_gurobi(self.robot_rvars[k], 1) for k in range(self.nbzs)]
        self.robot_dhvars = [get_derivative_control_points_gurobi(self.robot_hvars[k], 1) for k in range(self.nbzs)]
        
        
        ###### For plotting #######

        #self.obj_hvars[obj_pre_idx+1][0,0] = self.sol_robot_hvars[1][0,0]
        # We have changed the beginning of the inter curve (time of impact) and the end of the inter curve.
        # We let the object keep thinking thats its pre curve ends at the planned time
        # We then extend it inter curve to the new end time
        self.obj_hvars[obj_pre_idx+1][0,-1] = self.sol_robot_hvars[1][0,-1]

        # update the position after the inter curve
        self.obj_rvars[obj_pre_idx +2][:,0] = self.sol_robot_rvars[1][:,-1] + (self.rob_rad + self.obj_rad) * unit_push_dir
        
        #self.obj_hvars[obj_pre_idx +2][0,0] = self.sol_robot_hvars[1][0,-1]

        # get the speed at the end of the interaction curve
        self.obj_drvars[obj_pre_idx +2][:,0] = self.robot_drvars[pre_idx+1][:,-1]
        self.obj_dhvars[obj_pre_idx +2][0,0] = self.robot_dhvars[pre_idx+1][0,-1]

        self.obj_rvars[obj_pre_idx+2][:,1] = self.obj_rvars[obj_pre_idx+2][:,0] + self.obj_drvars[obj_pre_idx +2][:,0] / self.obj_dhvars[obj_pre_idx +2][0,0] * (self.obj_hvars[obj_pre_idx +2][0,1] - self.obj_hvars[obj_pre_idx +2][0,0])


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
            if 'inter' in self.robot_idvars[bz]:
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
            if 'inter' in self.robot_idvars[bz]:
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
            if 'inter' in self.robot_idvars[bz]:
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
            if 'inter' in self.robot_idvars[bz]:
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
        plt.savefig("/home/arian/repos/thesis/impact_stl/online_planner/figures/plot.svg")

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
        plt.savefig("/home/arian/repos/thesis/impact_stl/online_planner/figures/xy_plane.svg")

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
        #If we change the end time of the interation and then propogate this change to the rest of the curves, the final time of the plan changes
        # and no longer coincides with world end time. So we need to account for this in the animation time range
        self.t_range = np.linspace(0,self.world_tf+self.end_time_diff,Neval)

        self.fig_anim = plt.figure(figsize=(10,10))
        self.ax_anim = plt.axes()
        anim = animation.FuncAnimation(self.fig_anim,self._animate_update,
                                        frames=Neval,interval=self.world_tf/Neval*1e6*2)
        anim.save("/home/arian/repos/thesis/impact_stl/online_planner/figures/animation.mp4",writer="ffmpeg", fps=Neval/self.world_tf)


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
        #self.ax_anim.text(
        #c[0], c[1] + 0.25,  # slightly above the circle
        #'RobTheRobot',
        #color='k', fontsize=12, ha='center', va='bottom', weight='bold',
        #bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
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
    def save_plan(self):
        from utilities.read_write_plan import plan_to_csv
        rvars = [np.vstack((self.robot_rvars[bz],np.zeros((1,self.ncp)))) for bz in range(self.nbzs)]
        plan_to_csv(rvars, self.robot_hvars, self.robot_idvars, self.robot_other_names,
                    scenario_name="minimal_test_throw_and_catch",
                    robot_name = 'snap', 
                    path = '/home/arian/repos/thesis/impact_stl/online_planner/replans')
test = TestOnlinePlanner()
