#!/usr/bin/env python
__author__ = "Joris Verhagen"
__contact__ = "jorisv@kth.se"

import rclpy
import numpy as np
import time
import os

from planner.utilities.beziers import value_bezier, eval_t, get_derivative_control_points_gurobi
from impact_stl.helpers.read_write_plan import csv_to_plan

from rclpy.node import Node
from rclpy.clock import Clock
from impact_stl.helpers.qos_profiles import NORMAL_QOS, RELIABLE_QOS, RELIABLE_QOS_2

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleAngularVelocity
from px4_msgs.msg import VehicleAngularVelocity
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleRatesSetpoint

from my_msgs.srv import SetPlan, SetVerbosePlan
from my_msgs.msg import StampedBool, VerboseBezierPlan, TimeShift, Replan

from ament_index_python.packages import get_package_share_directory

from impact_stl.models.spacecraft_rate_model import SpacecraftRateModel
from impact_stl.planners.main_planner import plan_to_plan_msg
# from impact_stl.controller.rate_mpc import SpacecraftRateMPC
from impact_stl.controllers.rate_mpc_acados import SpacecraftRateMPC

from impact_stl.helpers.helpers import vector2PoseMsg, BezierCurve2NumpyArray, \
                            BezierPlan2NumpyArray, interpolate_bezier, VerboseBezierPlan2NumpyArray,\
                            Quaternion2Euler, Euler2Quaternion

class SpacecraftImpactMPC(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.get_logger().info('Creating SpacecraftImpactMPC node')
        # the object is an actual robot, so it has a namespace that we need 
        # for properly timing the replanning
        self.robot_name = self.get_namespace()
        self.object_ns = self.declare_parameter('object_ns', '/crackle').value
        self.scenario_name = self.declare_parameter('scenario_name', 'throw_and_catch').value
        self.enable_cbf =self.declare_parameter('enable_cbf', False).value
        self.get_logger().info(f"robot_name: {self.robot_name}, object_ns: {self.object_ns}, enable_cbf: {self.enable_cbf}")
        self.gz = self.declare_parameter('gz', True).value
        gz_suffix = '_gz' if self.gz else ''

        # get initial state from passed parameters
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            1.0, 0.0, 0.0, 0.0]).reshape(10, 1)
        self.x0[0] = self.declare_parameter('x0', 0.0).value
        self.x0[1] = self.declare_parameter('y0', 0.0).value
        self.x0[2] = self.declare_parameter('z0', 0.0).value
        self.x0[3] = self.declare_parameter('vx0', 0.0).value
        self.x0[4] = self.declare_parameter('vy0', 0.0).value
        self.x0[5] = self.declare_parameter('vz0', 0.0).value


        # Subscribers
        self.status_sub = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status',
            self.vehicle_status_callback,
            NORMAL_QOS)
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            'fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            NORMAL_QOS)
        self.angular_vel_sub = self.create_subscription(
            VehicleAngularVelocity,
            'fmu/out/vehicle_angular_velocity',
            self.vehicle_angular_velocity_callback,
            NORMAL_QOS)
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            'fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            NORMAL_QOS)
        
        # subscriber for the object for the CBF
        self.object_local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            f'{self.object_ns}/fmu/out/vehicle_local_position{gz_suffix}',
            self.object_local_position_callback,
            NORMAL_QOS)
        
        # Bezier planner stuff
        self.execute_plan_sub = self.create_subscription(
            StampedBool,
            'impact_stl/execute_plan',
            self.execute_plan_callback,
            RELIABLE_QOS)


        # Global reset detector
        self.global_reset_sub = self.create_subscription(
            StampedBool,
            '/global_reset',
            self.global_reset_callback,
            RELIABLE_QOS)


        self.global_time_shift_sub = self.create_subscription(
            TimeShift,
            '/global/impact_stl/time_shift',
            self.global_time_shift_callback,
            RELIABLE_QOS_2)
        
        # get the plan of the object from the csv file
        # this plan also never changes, so we can load it and use it forever :)
        package_share_directory = get_package_share_directory('impact_stl')
        plans_path = os.path.join(package_share_directory)
        try:
            self.get_logger().info(f"getting the plan for object {self.object_ns}")
            rvar,hvar,ids,other_names = csv_to_plan(robot_name=self.object_ns,
                                                    scenario_name=self.scenario_name,
                                                    path=plans_path)
            self.plan_object = VerboseBezierPlan2NumpyArray(plan_to_plan_msg(rvar,hvar,ids,other_names))
            # print some info
            self.get_logger().info(f"Number of bezier segments: {len(self.plan_object['rvar'])}")
            self.get_logger().info(f"Number of control points: {self.plan_object['rvar'][0].shape[1]}")
            self.get_logger().info(f"Segment ids: {self.plan_object['ids']}")
        except Exception as e:
            print(f"Could not find plan for object {self.object_ns}")
            print(f"Error: {e}")
            self.plan_object = None
        
        # service for obtaining an initial and an updated motion plan!
        # because this one can change, we can't just read the plan once and save it
        self.set_plan_srv = self.create_service(SetVerbosePlan, 'set_plan', self.add_set_plan_callback)
        self.start_time = 0     # Time when the plan starts, used to interpolate the Bezier curves to get the plan
        self.started = False    # Flag to start the plan, based on start_plan bool topic
        self.plan = None
        self.replanned = False  # Flag to start replanning when we arrive at pre-impact Bezier

        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', NORMAL_QOS)
        self.publisher_rates_setpoint = self.create_publisher(VehicleRatesSetpoint, 'fmu/in/vehicle_rates_setpoint', NORMAL_QOS)
        
        #self.predicted_path_pub = self.create_publisher(Path, 'impact_stl/predicted_path', 10)
        #self.reference_path_pub = self.create_publisher(Path, "impact_stl/reference_path", 10)
        
        self.entire_path_pub = self.create_publisher(Path, "impact_stl/entire_path", 10)
        self.replanned_path_pub = self.create_publisher(Path, "impact_stl/replanned_path", 10)
        
        self.publisher_recompute_local_plan = self.create_publisher(Replan, 'impact_stl/recompute_local_plan', RELIABLE_QOS)
        self.timer_period = 0.05  # seconds
        self.timer = self.create_timer(self.timer_period, self.cmdloop_callback)

        self.timer_period2 = 0.1
        self.offboard_timer = self.create_timer(self.timer_period2, self.offboard_callback)
        
        self.cooldown_replanner = 0.1  # seconds
        self.last_replan_time = -np.inf
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX

        # Create Spacecraft and controller objects
        self.model = SpacecraftRateModel()
        self.mpc = SpacecraftRateMPC(self.model,Tf=1.0,N=10,add_cbf=self.enable_cbf) # N = 10 for rape_mpc, 100 for rate_mpc_acados
        self.initial_guess = {'X': None, 'U': None}

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.setpoint_position = np.array([0.0, 0.0, 0.0])

        self.object_local_position = np.array([0.0, 0.0, 0.0])
        self.object_local_velocity = np.array([0.0, 0.0, 0.0])

    def global_time_shift_callback(self, msg):
        """Callback for global time shift messages.
        Args:
            msg (TimeShift): Message containing the time shift information.
        Effects:
            Shifts the timing of the robot's and object's plans if the message is not from this robot.
            The shift represents much to alter the timing of the plans to stay synchronized with global time changes.
            Time shifts are caused by the replanner, if we we are replanning our  plans from the replanner already contain the time shift.
            But if another robot has replanned, we need to shift our plans accordingly.
            This is done so we delay/avance our plans to stay in sync with the other robots.
            Its also done so that we know when our next interaction with the object is, to properly replan before that.
        """
        if msg.robot_name != self.robot_name:
            #Propogate time shift for robot plan and object plan
            #self.get_logger().info(f'Global time shift received: {msg.time_shift} seconds')

            # For all control points in the plan, if the time is after the current time, shift it
            for idx in range(len(self.plan['hvar'])):
                for cp in range(self.plan['hvar'][idx].shape[1]):
                    if self.plan['hvar'][idx][0,cp] > (Clock().now().nanoseconds/ 1e3 - self.start_time)/1e6:
                        self.plan['hvar'][idx][0,cp] += msg.time_shift
            
            # Make sure to recompute the dhvars so it fits the new timing
            self.plan['dhvar'] = [get_derivative_control_points_gurobi(hvar,1) for hvar in self.plan['hvar']]

            # Also shift the object plan, here we don't care about current time or dhvars, just the timing
            for idx in range(len(self.plan_object['hvar'])):
                self.plan_object['hvar'][idx][0,:] += msg.time_shift


    def global_reset_callback(self, msg):
        self.get_logger().info('Global reset received')
        self.started = False
        self.replanned = False
        self.t_object_coming = np.inf

    def vehicle_attitude_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_attitude[0] = msg.q[0]
        self.vehicle_attitude[1] = msg.q[1]
        self.vehicle_attitude[2] = -msg.q[2]
        self.vehicle_attitude[3] = -msg.q[3]

    def vehicle_local_position_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_local_position[0] = msg.x
        self.vehicle_local_position[1] = -msg.y
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vx
        self.vehicle_local_velocity[1] = -msg.vy
        self.vehicle_local_velocity[2] = -msg.vz

    def object_local_position_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.object_local_position[0] = msg.x
        self.object_local_position[1] = -msg.y
        self.object_local_position[2] = -msg.z
        self.object_local_velocity[0] = msg.vx
        self.object_local_velocity[1] = -msg.vy
        self.object_local_velocity[2] = -msg.vz

    def vehicle_angular_velocity_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_angular_velocity[0] = msg.xyz[0]
        self.vehicle_angular_velocity[1] = -msg.xyz[1]
        self.vehicle_angular_velocity[2] = -msg.xyz[2]

    def vehicle_status_callback(self, msg):
        # print("NAV_STATUS: ", msg.nav_state)
        # print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state


    def publish_rate_setpoint(self, u_pred):
        thrust_rates = u_pred[:, 0]
        thrust_command = thrust_rates[0:3]
        rates_setpoint_msg = VehicleRatesSetpoint()
        rates_setpoint_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        rates_setpoint_msg.roll  = float(thrust_rates[3])
        rates_setpoint_msg.pitch = -float(thrust_rates[4])
        rates_setpoint_msg.yaw   = -float(thrust_rates[5])
        rates_setpoint_msg.thrust_body[0] = float(thrust_command[0])
        rates_setpoint_msg.thrust_body[1] = -float(thrust_command[1])
        rates_setpoint_msg.thrust_body[2] = -float(thrust_command[2])
        self.publisher_rates_setpoint.publish(rates_setpoint_msg)


    def get_object_next_inter(self,t):
        """
        Get the index of the next interaction bezier segment of the object.
        Args:
            t: current time
        Returns:
            inter_idx: index of the next interaction bezier segment of the object
        """
        try:
            # Get all indices where the id is 'inter'
            #inter_indices = [i for i, x in enumerate(self.plan_object['ids']) if x == 'inter']
            inter_indices = np.where(np.array(self.plan_object['ids']) == 'inter')[0]
            # Get the end times of all interaction beziers
            inter_tEnd = [self.plan_object['hvar'][inter_idx][0,-1] for inter_idx in inter_indices]
            # The next interaction is either ongoing or in the future, so we find the first for which tStop > t
            inter_idx = next((inter_indices[i] for i,tend in enumerate(inter_tEnd) if tend > t), -1)
        except:
            inter_idx = -1
        return inter_idx

    def get_setpoints(self,dist):
        #current time in seconds, compared to start_time
        t = (Clock().now().nanoseconds / 1000 - self.start_time) / 1e6
        setpoints = []
        selectors = []
        tI = None
        weights = {'Q': None, 'Q_e': None, 'R': None}

        # if we haven't started the simulation, we just keep position at the setpoint
        if not self.started:
            setpoint = self.x0
            # print(f"setpoint: {setpoint.T}")
            # self.get_logger().info(f"setpoint in get_setpoints: {setpoint}")
            for i in range(self.mpc.N+1):
                ti = t+i*self.mpc.dt
                setpoints.append(setpoint)
            
            return setpoints, None, weights, tI
        # we started the simulation
        else:
            # compute the nominal plan via interpolating the bezier curve
            for i in range(self.mpc.N+1):
                ti = t+i*self.mpc.dt
                plani = interpolate_bezier(self.plan,ti)

                setpoints.append(np.array([plani['q'][0], plani['q'][1], 0.0,
                                            plani['dq'][0], plani['dq'][1], 0.0,
                                            1.0, 0.0, 0.0, 0.0]).reshape(10,1))
                selectors.append(1 if plani['id']=='inter' else 0)
            
            # Check if no interaction on Horizon and next interaction is ours
            if all(selectors[i]==0 for i in range(len(selectors))) and self.plan_object['other_names'][self.get_object_next_inter(t)] in self.robot_name and dist > 0.5:
                #self.get_logger().info('Calling Replanning Service in ff_rate_mpc_impact')

                #Cooldown for the replanner so we don't spam it
                if Clock().now().nanoseconds/1e9 - self.last_replan_time > self.cooldown_replanner:
                    self.last_replan_time = Clock().now().nanoseconds/1e9
                    msg = Replan()
                    msg.starttime = int(self.start_time)
                    msg.robot_plan = plan_to_plan_msg(self.plan['rvar'], self.plan['hvar'], self.plan['ids'], self.plan['other_names'])
                    msg.object_plan = plan_to_plan_msg(self.plan_object['rvar'], self.plan_object['hvar'], self.plan_object['ids'], self.plan_object['other_names'])
                    #self.get_logger().info(f"next interaction index of the object: {self.get_object_next_inter(t)}")
                    #self.get_logger().info(f"type of next interaction index of the object: {type(self.get_object_next_inter(t))}")
                    self.publisher_recompute_local_plan.publish(msg)


            #Checking if we are on an interaction and the end of the interaction is on the horizon
            # NOTE: WE might need to put some sort of delay after the interaction is done,
                # since the replanner won't change the pos directly after the interaction
                # so the MPC might slam into the newly released object.

            if any(selectors[i] ==1 for i in range(len(selectors)-1)) and selectors[-1]==0:

                # Make all the setpoints after the interaction equal to the last interaction setpoint
                # We do this so the MPC focuses on the interaction and not on going to some random place afterwards
                end_idx = next(i for i in range(len(selectors)) if selectors[i]==0)
                for point in range(end_idx, len(setpoints)):
                    setpoints[point] = setpoints[end_idx-1]

            return setpoints, selectors, weights, tI


    def offboard_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.direct_actuator = False
        offboard_msg.body_rate = True   # rate control
        self.publisher_offboard_mode.publish(offboard_msg)

    def cmdloop_callback(self):
        t0 = time.time()
        x0 = np.array([self.vehicle_local_position[0], self.vehicle_local_position[1], self.vehicle_local_position[2],
                       self.vehicle_local_velocity[0], self.vehicle_local_velocity[1], self.vehicle_local_velocity[2],
                       self.vehicle_attitude[0], self.vehicle_attitude[1], self.vehicle_attitude[2], self.vehicle_attitude[3]]).reshape(10, 1)
        xobj = np.array([self.object_local_position[0], self.object_local_position[1], self.object_local_position[2],
                         self.object_local_velocity[0], self.object_local_velocity[1], self.object_local_velocity[2],
                         1,0,0,0]).reshape(10, 1)
        # print(f"x0: {x0}")
        #Don't want to replan if we are too close to the object already
        dist = np.linalg.norm(self.vehicle_local_position[0:3] - self.object_local_position[0:3])
        # get the reference states and corresponding times in the horizon
        setpoints, selectors, weights, tI = self.get_setpoints(dist)
        # self.get_logger().info(f"setpoints: {setpoints[0][0:2].T}, pos: {self.vehicle_local_position[0:2].T}")
        #self.get_logger().info(f"selectors: {selectors}")
        # solve the mpc

        # now the initial guess has to be x_pred again, but check the size!
        if self.initial_guess['X'] is not None:
            self.initial_guess['X'] = self.initial_guess['X'][:,1::] if self.initial_guess['X'].shape[1] > self.mpc.N+1 else self.initial_guess['X']    
            self.initial_guess['U'] = self.initial_guess['U'][:,1::] if self.initial_guess['U'].shape[1] > self.mpc.N else self.initial_guess['U']
    
        enable_cbf = False
        if selectors is not None and self.enable_cbf:
            if all(s == 0 for s in selectors):
                enable_cbf = True
        x_pred, u_pred = self.mpc.solve(x0,setpoints,
                                        weights=weights,
                                        initial_guess=self.initial_guess,
                                        xobj=xobj,enable_cbf=enable_cbf,
                                        logger=self.get_logger(),
                                        verbose=False,selectors=selectors)
        
        self.initial_guess = {'X': x_pred, 'U': u_pred}

        # predicted_path_msg = Path()
        # for idx in range(x_pred.shape[1]):
        #     # print(f"idx: {idx}, x_pred: {x_pred[:,idx]}")
        #     predicted_state = x_pred[:,idx]
        #     # Publish time history of the vehicle path
        #     predicted_pose_msg = vector2PoseMsg('map', predicted_state[0:3], np.array([1.0, 0.0, 0.0, 0.0]))
        #     predicted_path_msg.header = predicted_pose_msg.header
        #     predicted_path_msg.poses.append(predicted_pose_msg)
        # self.predicted_path_pub.publish(predicted_path_msg)

        # setpoint_path_msg = Path()
        # for idx in range(len(setpoints)):
        #     setpoint = setpoints[idx]
        #     # print(f"setpoint[{idx}]: {setpoint[0:3]}")
        #     # Publish time history of the vehicle path
        #     setpoint_pose_msg = vector2PoseMsg('map', setpoint[0:3], np.array([1.0, 0.0, 0.0, 0.0]))
        #     setpoint_path_msg.header = setpoint_pose_msg.header
        #     setpoint_path_msg.poses.append(setpoint_pose_msg)
        # self.reference_path_pub.publish(setpoint_path_msg)

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_rate_setpoint(u_pred)
        
        #print(f"Time elapsed: {time.time() - t0}")
        if time.time() - t0 > self.timer_period:
            self.get_logger().info(f"LOOP TOOK TOO LONG: {time.time() - t0} (timer_period: {self.timer_period})")
            
    def add_set_plan_callback(self, request, response):
        #Save the robot plan
        self.plan = VerboseBezierPlan2NumpyArray(request.plan)
        
        if request.replanned:
            #Save the object plan with new times
            self.plan_object = VerboseBezierPlan2NumpyArray(request.object_plan)
        self.publish_plan(request.replanned)
        return response
    
    def publish_plan(self,replanned):
        if not replanned:
            start = 0
            stop = -1
            pub = self.entire_path_pub
            path_id = 'entire'
        else:
            t = (Clock().now().nanoseconds / 1000 - self.start_time) / 1e6
            start = self.get_pre_idx(t)
            stop = start +1
            pub = self.replanned_path_pub
            path_id = 'replanned'

        try:
            N = 100
            ts = np.linspace(self.plan['hvar'][start][0,0],self.plan['hvar'][stop][0,-1],N)
            path_msg = Path()
            for t in ts:
                plani = interpolate_bezier(self.plan,t)
                posei = vector2PoseMsg('world', np.array([plani['q'][0], plani['q'][1], -0.01]), np.array([1.0, 0.0, 0.0, 0.0]))
                path_msg.header = posei.header
                path_msg.poses.append(posei)
            #self.get_logger().info(f'Publishing the {path_id} path')
            pub.publish(path_msg)
        except Exception as e:
            self.get_logger().info(f"Could not publish the {path_id} path: {e}")
        

    def execute_plan_callback(self, msg):
        self.get_logger().info('Starting executing the plan')
        # check if self.plan is set
        if msg.data:
            if self.plan is None:
                self.get_logger().info('No plan set, cannot execute')
                self.started = False
            else:
                self.started = True
                self.start_time = Clock().now().nanoseconds / 1000
        
    def get_pre_idx(self,t):
        """
        Get the index of the next pre-impact bezier segment and the impact time tI.
        Args:
            t: current time
        Returns:
            pre_idx: index of the next pre-impact bezier segment
        """
        try:
            # Get all indices where the id is 'pre'
            pre_indices = [i for i, x in enumerate(self.plan['ids']) if x == 'pre']
            # Get the end times of all pre-impact beziers
            pre_tIs = [self.plan['hvar'][i][0,-1] for i in pre_indices]
            # impacts may only occur in the future, so we find the first for which tI > t
            pre_idx = next((pre_indices[i] for i, tI in enumerate(pre_tIs) if tI > t), len(pre_tIs)-1)
        except:
            pre_idx = -1
        return pre_idx

def main(args=None):
    rclpy.init(args=args)
    spacecraft_mpc = SpacecraftImpactMPC()
    rclpy.spin(spacecraft_mpc)

    spacecraft_mpc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
