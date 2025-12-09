#!/usr/bin/env python
__author__ = "Arian Kourangi"
__contact__ = "arianke@kth.se"

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
from impact_stl.controllers.rate_mpc_velocity_keeping import SpacecraftRateMPC

from impact_stl.helpers.helpers import vector2PoseMsg, BezierCurve2NumpyArray, \
                            BezierPlan2NumpyArray, interpolate_bezier, VerboseBezierPlan2NumpyArray,\
                            Quaternion2Euler, Euler2Quaternion

INTER_HOLD_TIME = 0.5  # seconds
class SpacecraftImpactMPC(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.get_logger().info('Creating SpacecraftImpactMPC node')
        # the object is an actual robot, so it has a namespace that we need 
        # for properly timing the replanning

        self.robot_name = self.get_namespace()
        self.get_logger().info(f"robot_name: {self.robot_name}: Velocity Keeping MPC Controller Node started")
        self.hw = self.declare_parameter('hw', False).value
        gz_suffix = '' if self.hw else '_gz'
        # get initial state from passed parameters
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            1.0, 0.0, 0.0, 0.0]).reshape(10, 1)
        self.x0[0] = self.declare_parameter('x0', 0.0).value
        self.x0[1] = self.declare_parameter('y0', 0.0).value
        self.x0[2] = self.declare_parameter('z0', 0.0).value
        self.x0[3] = self.declare_parameter('vx0', 0.0).value
        self.x0[4] = self.declare_parameter('vy0', 0.0).value
        self.x0[5] = self.declare_parameter('vz0', 0.0).value

        self.started = False
        self.saved = False
        self.last_inter_time = np.array([-np.inf])
        # Subscribers
        self.status_sub = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status',
            self.vehicle_status_callback,
            NORMAL_QOS)
        self.status_sub = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            NORMAL_QOS)
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            'fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            NORMAL_QOS)
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            f'fmu/out/vehicle_local_position{gz_suffix}',
            self.vehicle_local_position_callback,
            NORMAL_QOS)
        
        self.inter_sub = self.create_subscription(
            StampedBool,
            '/global/interaction',
            self.interaction_callback,
            RELIABLE_QOS_2)
        
        # Create Spacecraft and controller objects
        self.model = SpacecraftRateModel()
        self.mpc = SpacecraftRateMPC(self.model,Tf=1.0,N=10) 
        self.initial_guess = {'X': None, 'U': None}

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_time = np.array([0.0])
        self.release_position = np.array([0.0, 0.0, 0.0])
        self.setpoint_position = np.array([0.0, 0.0, 0.0])

        self.object_local_position = np.array([0.0, 0.0, 0.0])
        self.object_local_velocity = np.array([0.0, 0.0, 0.0])
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX

        
        self.publisher_rates_setpoint = self.create_publisher(VehicleRatesSetpoint, 'fmu/in/vehicle_rates_setpoint', NORMAL_QOS)
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', NORMAL_QOS)
        self.timer_period2 = 0.1
        self.offboard_timer = self.create_timer(self.timer_period2, self.offboard_callback)
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.cmdloop_callback)
        self.release_time = np.array([0.0])

    def interaction_callback(self, msg):
        self.last_inter_time = self.vehicle_local_time.copy()
        #self.v_des = self.vehicle_local_velocity.copy()
        #self.release_position = self.vehicle_local_position.copy()
        self.started = True
        self.saved = False
        #print('Last_interaction_time',self.last_inter_time)
    def vehicle_attitude_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_attitude = self.q_ned_to_q_enu(np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]]))

    def q_ned_to_q_enu(self, q_ned):
        # Convert NED quaternion to ENU quaternion
        # q is in the form (qw, qx, qy, qz) and describes the rotation from body frame to global frame
        # Yes, NED <-> ENU  is symmetric
        q_enu = 1/np.sqrt(2) * np.array([q_ned[0] + q_ned[3], q_ned[1] + q_ned[2], q_ned[1] - q_ned[2], q_ned[0] - q_ned[3]])
        q_enu /= np.linalg.norm(q_enu)
        return q_enu.astype(float)


    def vehicle_local_position_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_local_time[0] = msg.timestamp / 1e6
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz


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
        # Might ened to switch these for right frame conversion
        k = 2.8 if self.hw else 1.0
        rates_setpoint_msg.thrust_body[0] = float(thrust_command[0])/k
        rates_setpoint_msg.thrust_body[1] = -float(thrust_command[1])/k
        rates_setpoint_msg.thrust_body[2] = -float(thrust_command[2])/k
        self.publisher_rates_setpoint.publish(rates_setpoint_msg)


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

        if self.initial_guess['X'] is not None:
            self.initial_guess['X'] = self.initial_guess['X'][:,1::] if self.initial_guess['X'].shape[1] > self.mpc.N+1 else self.initial_guess['X']    
            self.initial_guess['U'] = self.initial_guess['U'][:,1::] if self.initial_guess['U'].shape[1] > self.mpc.N else self.initial_guess['U']
        
        setpoints = []
        t = Clock().now().nanoseconds  / 1e9

        if not self.started:
            setpoint = self.x0
            for i in range(self.mpc.N+1):
                ti = t+i*self.mpc.dt
                setpoints.append(setpoint)
            self.v_des = np.array([0.0, 0.0, 0.0])
        elif t - self.last_inter_time[0] > INTER_HOLD_TIME:
            if not self.saved:  
                self.v_des = self.vehicle_local_velocity.copy()
                self.release_position = self.vehicle_local_position.copy()
                self.release_time = self.vehicle_local_time.copy()
                self.saved = True

            for i in range(self.mpc.N+1):
                ti = t+i*self.mpc.dt
                setpoint = np.zeros((10,1))
                # position
                setpoint[0] = self.release_position[0] + (ti - self.release_time[0])*self.v_des[0]
                setpoint[1] = self.release_position[1] + (ti - self.release_time[0])*self.v_des[1]
                setpoint[2] = 0.0 
                # velocity
                setpoint[3] = self.v_des[0]
                setpoint[4] = self.v_des[1]
                setpoint[5] = 0.0
                # attitude
                setpoint[6] = 1.0
                setpoint[7] = 0.0
                setpoint[8] = 0.0
                setpoint[9] = 0.0
                setpoints.append(setpoint)
        
        if t - self.last_inter_time[0] > INTER_HOLD_TIME:
        #print('self.started',self.started)
        #if not self.started:
            # We just interacted, need to keep the velocity we had after the interaction to combat friciton of the floor
            x_pred, u_pred ,status= self.mpc.solve(x0,setpoints,
                                            initial_guess=self.initial_guess)
            if status == 0:
                self.initial_guess = {'X': x_pred, 'U': u_pred}
        

            if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and status == 0:
                self.publish_rate_setpoint(u_pred)

            #print(f"Time elapsed: {time.time() - t0}")
            if time.time() - t0 > self.timer_period:
                self.get_logger().info(f"LOOP TOOK TOO LONG: {time.time() - t0} (timer_period: {self.timer_period})")
        else:
            self.initial_guess = {'X': None, 'U': None}
            self.publish_rate_setpoint(np.zeros((6,1)))
        #print('v_des',self.v_des)
def main(args=None):
    rclpy.init(args=args)
    spacecraft_mpc = SpacecraftImpactMPC()
    rclpy.spin(spacecraft_mpc)

    spacecraft_mpc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
