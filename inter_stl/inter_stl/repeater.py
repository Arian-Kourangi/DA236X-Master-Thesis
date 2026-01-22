#!/usr/bin/env python
__author__ = "Arian Kourangi"
__contact__ = "arianke@kth.se"

import rclpy
import numpy as np
import time
import os

from planner.utilities.beziers import value_bezier, eval_t, get_derivative_control_points_gurobi
from inter_stl.helpers.read_write_plan import csv_to_plan
from rclpy.node import Node
from rclpy.clock import Clock
from inter_stl.helpers.qos_profiles import NORMAL_QOS, RELIABLE_QOS, RELIABLE_QOS_2

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


from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleRatesSetpoint
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from my_msgs.msg import StampedBool

def vector2PoseMsg(frame_id, position, attitude):
    pose_msg = PoseStamped()
    # msg.header.stamp = Clock().now().nanoseconds / 1000
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.orientation.w = attitude[0]
    pose_msg.pose.orientation.x = attitude[1]
    pose_msg.pose.orientation.y = attitude[2]
    pose_msg.pose.orientation.z = attitude[3]
    pose_msg.pose.position.x = position[0]
    pose_msg.pose.position.y = position[1]
    pose_msg.pose.position.z = 0*position[2]
    return pose_msg


class Repeater(Node):

    def __init__(self):
        super().__init__('repeater')
        self.name = self.get_namespace().replace("/", "")
        qos_profile = NORMAL_QOS

        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            "fmu/out/vehicle_local_position",
            self.vehicle_local_position_callback,
            qos_profile,
        )
        
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            "fmu/out/vehicle_attitude",
            self.vehicle_attitude_callback,
            qos_profile,
        )
        
        self.vehicle_pose_pub = self.create_publisher(
            PoseStamped, "px4_visualizer/vehicle_pose", 10
        )
        self.vehicle_vel_pub = self.create_publisher(
            Marker, "px4_visualizer/vehicle_velocity", 10)
        self.vehicle_path_pub = self.create_publisher(
            Path, "arian/vehicle_path", 10
        )
        # self.setpoint_path_pub = self.create_publisher(
        #     Path, "px4_visualizer/setpoint_path", 10
        # )
        self.vehicle_name_pub = self.create_publisher(
            Marker, "arian/vehicle_name", 10)
        
        # self.vehicle_force_pub = self.create_publisher(
        #     Marker, "px4_visualizer/vehicle_force", 10
        # )




        # trail size
        self.trail_size = 100

        # time stamp for the last local position update received on ROS2 topic
        self.last_local_pos_update = 0.0
        
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.setpoint_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_force = np.array([0.0, 0.0, 0.0])
        self.vehicle_path_msg = Path()
        self.setpoint_path_msg = Path()

        self.declare_parameter("path_clearing_timeout", 1.0)
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)



    def vehicle_attitude_callback(self, msg):
        # TODO: handle NED->ENU transformation
        #self.vehicle_attitude[0] = msg.q[0]
        #self.vehicle_attitude[1] = msg.q[1]
        #self.vehicle_attitude[2] = -msg.q[2]
        #self.vehicle_attitude[3] = -msg.q[3]
        self.vehicle_attitude = self.q_ned_to_q_enu(np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]]))

    def q_ned_to_q_enu(self, q_ned):
        # Convert NED quaternion to ENU quaternion
        # q is in the form (qw, qx, qy, qz) and describes the rotation from body frame to global frame
        # Yes, NED <-> ENU  is symmetric
        q_enu = 1/np.sqrt(2) * np.array([q_ned[0] + q_ned[3], q_ned[1] + q_ned[2], q_ned[1] - q_ned[2], q_ned[0] - q_ned[3]])
        q_enu /= np.linalg.norm(q_enu)
        return q_enu.astype(float)
    

    def create_robot_name_marker(self, id, position):
        name_list = {'pop': 'Robot_1', 'crackle': 'Robot_2', 'snap': 'Object'}
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "world"
        #msg.header.stamp = Clock().now().nanoseconds / 1000
        msg.ns = "robot_name"
        msg.id = id
        msg.type = Marker.TEXT_VIEW_FACING
        # msg.scale.x = 2.0 * radius
        # msg.scale.y = 2.0 * radius
        msg.scale.z = 0.1

        msg.color.r = 1.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.color.a = 2.0
        msg.text = name_list[self.name]
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        return msg


    def vehicle_local_position_callback(self, msg):
        # print("!!! receiving vehicle_local_position message")
        path_clearing_timeout = (
            self.get_parameter("path_clearing_timeout")
            .get_parameter_value()
            .double_value
        )
        if path_clearing_timeout >= 0 and (
            (Clock().now().nanoseconds / 1e9 - self.last_local_pos_update)
            > path_clearing_timeout
        ):
            self.vehicle_path_msg.poses.clear()
        self.last_local_pos_update = Clock().now().nanoseconds / 1e9

        # TODO: handle NED->< transformation
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz


    def append_vehicle_path(self, msg):
        self.vehicle_path_msg.poses.append(msg)
        if len(self.vehicle_path_msg.poses) > self.trail_size:
            del self.vehicle_path_msg.poses[0]

    def append_setpoint_path(self, msg):
        self.setpoint_path_msg.poses.append(msg)
        if len(self.setpoint_path_msg.poses) > self.trail_size:
            del self.setpoint_path_msg.poses[0]

    def cmdloop_callback(self):
        vehicle_pose_msg = vector2PoseMsg(
            "world", self.vehicle_local_position, self.vehicle_attitude
        )
        self.vehicle_pose_pub.publish(vehicle_pose_msg)

        # Publish time history of the vehicle path
        self.vehicle_path_msg.header = vehicle_pose_msg.header
        self.append_vehicle_path(vehicle_pose_msg)
        self.vehicle_path_pub.publish(self.vehicle_path_msg)

        # Publish time history of the vehicle path
        # setpoint_pose_msg = vector2PoseMsg("world", self.setpoint_position, self.vehicle_attitude)
        # self.setpoint_path_msg.header = setpoint_pose_msg.header
        # self.append_setpoint_path(setpoint_pose_msg)
        # self.setpoint_path_pub.publish(self.setpoint_path_msg)

        # Publish arrow markers for velocity
        # velocity_msg = self.create_arrow_marker(1, self.vehicle_local_position, self.vehicle_local_velocity)
        # self.vehicle_vel_pub.publish(velocity_msg)

        # Create a circle marker with the vehicle radius
        vehicle_name_msg = self.create_robot_name_marker(1, self.vehicle_local_position)
        self.vehicle_name_pub.publish(vehicle_name_msg)

        # Create an arrow for the vehicle force
        # self.get_logger().info(f"Vehicle force: {self.vehicle_force}")
        # force_msg = self.create_arrow_marker(2, self.vehicle_local_position, self.vehicle_force)
        # self.vehicle_force_pub.publish(force_msg)

def main(args=None):
    rclpy.init(args=args)
    spacecraft_mpc = Repeater()
    rclpy.spin(spacecraft_mpc)

    spacecraft_mpc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
