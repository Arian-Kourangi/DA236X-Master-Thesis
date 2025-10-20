#!/usr/bin/env python
__author__ = "Joris Verhagen"
__contact__ = "jorisv@kth.se"

import numpy as np
import time
import rclpy
import casadi as cs
from rclpy.node import Node
from rclpy.clock import Clock
from impact_stl.helpers.qos_profiles import NORMAL_QOS, RELIABLE_QOS
import os
import cvxpy as cp
import copy

from px4_msgs.msg import VehicleAngularVelocity
from px4_msgs.msg import VehicleAngularVelocity
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleLocalPosition

from my_msgs.msg import StampedBool, Replan, TimeShift

from ament_index_python.packages import get_package_share_directory

from impact_stl.planners.main_planner import MinimalClientAsync
from impact_stl.helpers.beziers import get_derivative_control_points_gurobi, get_derivative_control_points_cvxpy
from impact_stl.helpers.read_write_plan import csv_to_plan, plan_to_csv
#from impact_stl.helpers.solve_two_body_impact import solve_two_body_impact
from impact_stl.helpers.plot_rvars_hvars import plot_rvars_hvars

from impact_stl.helpers.helpers import vector2PoseMsg, BezierCurve2NumpyArray, \
                            BezierPlan2NumpyArray, interpolate_bezier, VerboseBezierPlan2NumpyArray,\
                            Quaternion2Euler, Euler2Quaternion

class RePlanner(Node):
    def __init__(self):
        super().__init__('replanner')
        self.get_logger().info('Creating Replanner Node')

        self.minimal_client = MinimalClientAsync()

        # the object is an actual robot, so it has a namespace
        self.robot_name = self.get_namespace() if self.get_namespace() != '/' else 'pop'
        self.scenario_name = self.declare_parameter('scenario_name', 'throw_and_catch_exp').value
        self.object_ns = self.declare_parameter('object_ns', '/pop').value
        self.gz = self.declare_parameter('gz', True).value
        print(f"object_ns: {self.object_ns}")

        gz_suffix = '_gz' if self.gz else ''
        # object subscribers
        self.object_attitude_sub = self.create_subscription(
            VehicleAttitude,
            f'{self.object_ns}/fmu/out/vehicle_attitude{gz_suffix}',
            self.object_attitude_callback,
            NORMAL_QOS)
        self.object_angular_vel_sub = self.create_subscription(
            VehicleAngularVelocity,
            f'{self.object_ns}/fmu/out/vehicle_angular_velocity{gz_suffix}',
            self.object_angular_velocity_callback,
            NORMAL_QOS)
        self.object_local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            f'{self.object_ns}/fmu/out/vehicle_local_position{gz_suffix}',
            self.object_local_position_callback,
            NORMAL_QOS)
        
        # robot subscribers
        self.robot_attitude_sub = self.create_subscription(
            VehicleAttitude,
            'fmu/out/vehicle_attitude',
            self.robot_attitude_callback,
            NORMAL_QOS)
        self.robot_angular_vel_sub = self.create_subscription(
            VehicleAngularVelocity,
            'fmu/out/vehicle_angular_velocity',
            self.robot_angular_velocity_callback,
            NORMAL_QOS)
        self.robot_local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            'fmu/out/vehicle_local_position',
            self.robot_local_position_callback,
            NORMAL_QOS)
        
        # replan subscriber
        self.replan_sub = self.create_subscription(
            Replan,
            'impact_stl/recompute_local_plan',
            self.recompute_local_plan_callback,
            RELIABLE_QOS)
        self.time_shift_pub = self.create_publisher(
            TimeShift,
            '/global/impact_stl/time_shift',
            RELIABLE_QOS)
        ## get the original plan from the csv file
        #package_share_directory = get_package_share_directory('impact_stl')
        #plans_path = os.path.join(package_share_directory)
        #self.rvars,self.hvars,self.idvars,self.other_names = csv_to_plan(self.robot_name,
        #                                                                 scenario_name=self.scenario_name,
        #                                                                 path=plans_path)
        ## get the plan of the object from the csv file
        #try:
        #    self.orvars,self.ohvars,self.oidvars,self.oother_names = csv_to_plan(self.object_ns,
        #                                                                         scenario_name=self.scenario_name,
        #                                                                         path=plans_path)
        #except Exception as e:
        #    print(f"Could not find plan for object {self.object_ns}")
        #    print(f"Error: {e}")

        #self.drvars = [get_derivative_control_points_gurobi(rvar) for rvar in self.rvars]
        #self.dhvars = [get_derivative_control_points_gurobi(hvar) for hvar in self.hvars]
        
        # From the .sdf file:
        # /home/px4space/PX4/PX4-Space-Systems/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/2d_spacecraft/2d_spacecraft.sdf
        self.rob_rad = 0.20
        self.obj_rad = 0.20
        self.end_time_diff = 0.0
        self.old_robot_hvars = None
        self.inter_id = 0
        #Size of world
        self.world_lb = np.array([0,0])
        self.world_ub = np.array([10,20])
        self.dq_lb = np.array([-2,-2])
        self.dq_ub = np.array([2,2])  

        # position and velocity variables that are updated with the subscriber calls
        self.object_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.object_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.object_local_position = np.array([0.0, 0.0, 0.0])
        self.object_local_velocity = np.array([0.0, 0.0, 0.0])
        # keep track of a stack of velocities to interpolate
        #TODO: if this is linked to an EKF we should use that, but the EKF needs to consider impacts
        stack_size = 15 # was 30
        self.start_time = 0
        self.object_local_velocity_stack = np.zeros((3,stack_size))
        self.object_local_position_stack = np.zeros((3,stack_size))
        self.object_local_position_time_stack = np.zeros((1,stack_size))

        self.robot_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.robot_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.robot_local_position = np.array([0.0, 0.0, 0.0])
        self.robot_local_velocity = np.array([0.0, 0.0, 0.0])

        self.Ncalls = 0
        self.Ncallbacks = 0
        self.verbose = True

        self.get_logger().info('Finished Replanner Node')

        
    def object_attitude_callback(self, msg):
        self.object_attitude[0] = msg.q[0]
        self.object_attitude[1] = msg.q[1]
        self.object_attitude[2] = -msg.q[2]
        self.object_attitude[3] = -msg.q[3]
    def robot_attitude_callback(self, msg):
        self.robot_attitude[0] = msg.q[0]
        self.robot_attitude[1] = msg.q[1]
        self.robot_attitude[2] = -msg.q[2]
        self.robot_attitude[3] = -msg.q[3]

    def object_angular_velocity_callback(self, msg):
        self.object_angular_velocity[0] = msg.xyz[0]
        self.object_angular_velocity[1] = -msg.xyz[1]
        self.object_angular_velocity[2] = -msg.xyz[2]
    def robot_angular_velocity_callback(self, msg):
        self.robot_angular_velocity[0] = msg.xyz[0]
        self.robot_angular_velocity[1] = -msg.xyz[1]
        self.robot_angular_velocity[2] = -msg.xyz[2]

    def object_local_position_callback(self, msg):
        self.object_local_position_time_stack[:,0:-1] = self.object_local_position_time_stack[:,1:]
        self.object_local_position_time_stack[:,-1] = msg.timestamp
        self.object_local_position[0] = msg.x
        self.object_local_position[1] = -msg.y
        self.object_local_position[2] = -msg.z
        self.object_local_position_stack[:,0:-1] = self.object_local_position_stack[:,1:]
        self.object_local_position_stack[:,-1] = self.object_local_position
        self.object_local_velocity[0] = msg.vx
        self.object_local_velocity[1] = -msg.vy
        self.object_local_velocity[2] = -msg.vz
        self.object_local_velocity_stack[:,0:-1] = self.object_local_velocity_stack[:,1:]
        self.object_local_velocity_stack[:,-1] = self.object_local_velocity

    def robot_local_position_callback(self, msg):
        self.robot_local_position[0] = msg.x
        self.robot_local_position[1] = -msg.y
        self.robot_local_position[2] = -msg.z
        self.robot_local_velocity[0] = msg.vx
        self.robot_local_velocity[1] = -msg.vy
        self.robot_local_velocity[2] = -msg.vz
        
    
    def recompute_local_plan_callback(self, msg):
        # based on the position and velocities of the robot and obstacle
        # plus the desired post-impact positions and velocities of the obstacle
        # recompute the pre- and post-impact Beziers to deal with the sizes
        self.get_logger().info('Recomputing local plan for pre- and post- impact Bezier')
        self.start_time = msg.starttime
        self.get_logger().info(f"Start time: {self.start_time}")
        robot_plan = VerboseBezierPlan2NumpyArray(msg.robot_plan)
        object_plan = VerboseBezierPlan2NumpyArray(msg.object_plan)

        # To not add more timeshifts when replanning the same inter multiple times
        # The inter_id is used if the robot has different interactions in the plan so it resets

        if self.old_robot_hvars is None or self.inter_id != msg.inter_id:
            self.old_robot_hvars = robot_plan['hvar']
            self.end_time_diff = 0.0
        elif msg.inter_id == self.inter_id:
            # If we get a replan for the same inter it can happen without the plans in mpc being updated, 
            # so to avoid time_shifts accumalating we save the hvars from the previous
            self.old_robot_hvars = copy.deepcopy(self.robot_hvars)

        self.inter_id = msg.inter_id

        self.robot_rvars = robot_plan['rvar']
        self.robot_hvars = robot_plan['hvar']
        self.robot_drvars = robot_plan['drvar']
        self.robot_dhvars = robot_plan['dhvar']
        self.robot_idvars = robot_plan['ids']
        self.robot_other_names = robot_plan['other_names']

        self.obj_rvars = object_plan['rvar']
        self.obj_hvars = object_plan['hvar']
        self.obj_drvars = object_plan['drvar']
        self.obj_dhvars = object_plan['dhvar']
        self.obj_idvars = object_plan['ids']
        self.obj_other_names = object_plan['other_names']

        ########### NOW SOLVE THE REPLANNING PROBLEM ############
        try:
            self.solve_replan_new()
        #self.get_logger().info("Local plan recomputed")
        except Exception as e:
            self.get_logger().error(f"Could not replan: {e}")
            return
        #Send out timeshift to all robots
        msg = TimeShift()
        msg.time_shift = float(self.end_time_diff)
        msg.robot_name = self.robot_name
        self.time_shift_pub.publish(msg)
        self.get_logger().info(f"Published time shift: {self.end_time_diff} seconds")
        #Sending the new plan to the ff_rate_mpc_impact node
        #self.get_logger().info('Sending plan')
        self.minimal_client.send_request(self.robot_rvars, self.robot_hvars, self.robot_idvars, self.robot_other_names,
                                       self.obj_rvars, self.obj_hvars, self.obj_idvars, self.obj_other_names)
        #self.get_logger().info('Plan received')
    
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

    def solve_replan_new(self):
        opti = cs.Opti()

        t_meas = (self.object_local_position_time_stack[0,-1] - self.start_time) / 1e6 # Convert from microseconds to seconds
        #self.get_logger().info(f"measured at time: {self.object_local_position_time_stack[0,-1]}")
        #self.get_logger().info(f"start_time: {self.start_time}")
        #self.get_logger().info(f"t_meas: {t_meas}")
        pos_meas = self.object_local_position[0:2]
        vel_meas = np.mean(self.object_local_velocity_stack,axis=1)[0:2]


        # Function to predict the position of the object at time t based on current measurement and constant velocity model
        predicted_pos = lambda t: pos_meas + vel_meas * (t - t_meas)
        n_cp = 6 # number of control points per curve


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
        x_start = self.robot_rvars[pre_idx][0:2,0]
        
        # This is the end position of the object at the end of the interaction curve, but because the offline planner assumes point masses the robots is the same
        #x_end = self.robot_rvars[pre_idx+1][:,-1]
        # I changed it to be the beginning of the curve after the inter curve, since it won't be affected by the replanning
        x_end = self.obj_rvars[obj_pre_idx+2][0:2,0]
        
        dr_start = self.robot_drvars[pre_idx][0:2,0] 
        dh_start = self.robot_dhvars[pre_idx][0,0]

        dr_end = self.robot_drvars[pre_idx+1][0:2,-1]
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
        next_obj_int_pos = self.obj_rvars[obj_next_pre][0:2,-1]
        #Vector between planned end of interaction and the next object interaction position
        v = next_obj_int_pos - x_end
        
        if v[0] == 0 and v[1] == 0:
            pass
        else:
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
            if x_pos:
                opti.subject_to(drvars[1][0,cp+1] - drvars[1][0,cp] >= 0) # increasing positive x velocity or decreasing negative x velocity
            else:
                opti.subject_to(drvars[1][0,cp+1] - drvars[1][0,cp] <= 0) # decreasing positive x velocity or increasing negative x velocity
            if y_pos:
                opti.subject_to(drvars[1][1,cp+1] - drvars[1][1,cp] >= 0) # increasing positive y velocity or decreasing negative y velocity
            else:
                opti.subject_to(drvars[1][1,cp+1] - drvars[1][1,cp] <= 0) # decreasing positive y velocity or increasing negative y velocity
            opti.subject_to(dhvars[1][0,cp+1] == dhvars[1][0,cp]) # Constant time derivative for now

        # Make sure the ratio of change in y velocity to change in x velocity is the same as the desired delta_V
        # This ensures that the robot pushes in the direction of desired velocity i.e the interaction angle is constant
        for i in range(drvars[1].shape[1]-1):
            d_x = drvars[1][0,cp+1] - drvars[1][0,cp]
            d_y = drvars[1][1,cp+1] - drvars[1][1,cp]
            opti.subject_to(d_x * delta_V[1] - d_y * delta_V[0] == 0) # cross product = 0 means they are colinear
        
        # Ensure paralelle final velocity, since dh is constant we can ignore it
        dq_end = dr_end/dh_end
        replanned_dq_end = drvars[1][:,-1]/dhvars[1][0,-1]
        if dq_end[0] == 0 and dq_end[1] == 0:
            pass
        else:
            opti.subject_to(dq_end[0] * replanned_dq_end[1] - dq_end[1] * replanned_dq_end[0] == 0) # cross product = 0 means they are colinear/parallel


        # Minimize the acceleration
        J = 0
        # For pre curve we can have smaller weights
        for i in range(ddrvars[0].shape[1]):
            J += cs.sumsqr(ddrvars[0][:,i])
        for i in range(ddhvars[0].shape[1]):
            J += cs.sumsqr(ddhvars[0][0,i])
        
        # For interaction curve we want to minimize acceleration more
        w_acc = 1e2
        for i in range(ddrvars[1].shape[1]):
            J += w_acc*cs.sumsqr(ddrvars[1][:,i])
        for i in range(ddhvars[1].shape[1]):
            J += w_acc*cs.sumsqr(ddhvars[1][0,i])
        
        # --- Soft penalties for final targets (relax exact equalities) ---
        # Weights (tune as needed)
        w_r = 1e5
        w_dr = 1e4
        w_h = 1e1
        w_dh = 1e4
        # We also want the end of the itneraction to be as close as possible to the planned end of interaction
        target_r_end = cs.DM(x_end)
        target_h_tf = float(tf)
        target_dr_end = cs.DM(dr_end)
        target_dh_end = float(dh_end)
        try:
            # target_* were prepared earlier (CasADi DM or floats)
            object_pos = rvars[1][:,-1]+(self.rob_rad + self.obj_rad) * unit_push_dir
            J += w_r * cs.sumsqr(object_pos- target_r_end)
            J += w_dr * cs.sumsqr(drvars[-1][:,-1] - target_dr_end)
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
            'polish':True}, 'warm_start_primal': True, 'warm_start_dual': True, 'error_on_fail': False
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

        sol = opti.solve()

        #Extract the solution
        self.sol_robot_rvars = [sol.value(rvars[k]).reshape(2,n_cp) for k in range(len(rvars))]
        self.sol_robot_hvars = [sol.value(hvars[k]).reshape(1,n_cp) for k in range(len(hvars))]
        #Add padding to make them compatible with the rest of the plan
        self.sol_robot_rvars = [np.vstack([rvar,np.zeros((1,rvar.shape[1]))]) for rvar in self.sol_robot_rvars]

        self.update_plan(pre_idx, obj_pre_idx)
        # I have to update the object plan as well since the interaction time might change and then for 
        # subsequent replans if I want to replan past the pre-computed objectplans impact time it will choose the wrong pre idx

        #Then the timeshift I will only apply it if its a different robot. I will apply to both robot and object

    def update_plan(self, pre_idx, obj_pre_idx):

        """
        Args:
            pre_idx (int): index of the first pre curve after t_meas
            obj_pre_idx (int): index of the first pre curve for the object after t_meas
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
        end_time_diff  = self.sol_robot_hvars[1][0,-1] - self.robot_hvars[pre_idx+1][0,-1]
        ## Propogate time change to the rest of the curves
        for k in range(pre_idx+2, len(self.robot_hvars)):
            self.robot_hvars[k][0,:] += end_time_diff
    
        self.robot_hvars[pre_idx] = self.sol_robot_hvars[0]
        self.robot_hvars[pre_idx+1] = self.sol_robot_hvars[1]

        # Propogate the time change to the rest of the object curves
        for k in range(obj_pre_idx+2, len(self.obj_hvars)):
            self.obj_hvars[k][0,:] += end_time_diff
        
        # Update the objects pre and inter cruves to have the same interaction and end time as the robots
        # NOTE: WE only update the time since we want to keep the original planned positions of the object, for subsequent replannings
        self.obj_hvars[obj_pre_idx][0,-1] = self.robot_hvars[pre_idx][0,-1]
        self.obj_hvars[obj_pre_idx+1][0,0] = self.robot_hvars[pre_idx+1][0,0]
        self.obj_hvars[obj_pre_idx+1][0,-1] = self.robot_hvars[pre_idx+1][0,-1]

        #For publishing the timeshift
        self.end_time_diff = self.sol_robot_hvars[1][0,-1] - self.old_robot_hvars[pre_idx+1][0,-1]


def main(args=None):
    rclpy.init(args=args)
    replanner = RePlanner()
    rclpy.spin(replanner)

    replanner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()