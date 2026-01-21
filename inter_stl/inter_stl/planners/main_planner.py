# ROS2 service that considers the position of a robot (snap) and a given target
# and computes a bezier curve trajectory that connects the two points.

import numpy as np
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import casadi as cs
import os

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from my_msgs.msg import BezierCurve, BezierPlan, StampedBool
from my_msgs.msg import VerboseBezierPlan
from my_msgs.srv import SetPlan, SetVerbosePlan
from px4_msgs.msg import VehicleLocalPosition
from inter_stl.helpers.beziers import get_derivative_control_points_gurobi
from ament_index_python.packages import get_package_share_directory
from inter_stl.helpers.read_write_plan import csv_to_plan

def plan_to_plan_msg(rvars,hvars,idvars,other_names):

    """
    Converts a list of rvars and hvars (numpy arrays) to a BezierPlan message.
    Args:
        rvars: list of numpy arrays of shape (3, n) where n is the number of control points
        hvars: list of numpy arrays of shape (1, n) where n is the
        idvars: list of strings
        other_names: list of strings
    returns: 
        BezierPlan message

    """
    nbzs = len(rvars)
    rvars = [rvar.astype(np.float64) for rvar in rvars]
    hvars = [hvar.astype(np.float64) for hvar in hvars]

    drvars = [get_derivative_control_points_gurobi(rvar) for rvar in rvars]
    dhvars = [get_derivative_control_points_gurobi(hvar) for hvar in hvars]

    plan = VerboseBezierPlan()
    plan.rvar = [BezierCurve() for _ in rvars]
    plan.hvar = [BezierCurve() for _ in hvars]
    plan.drvar = [BezierCurve() for _ in drvars]
    plan.dhvar = [BezierCurve() for _ in dhvars]
    plan.ids = [str() for _ in idvars]
    plan.other_names = [str() for _ in idvars]

    for i in range(nbzs):
        plan.rvar[i].x_cp = rvars[i][0,:].tolist()
        plan.rvar[i].y_cp = rvars[i][1,:].tolist()
        plan.rvar[i].z_cp = rvars[i][2,:].tolist()
        plan.drvar[i].x_cp = drvars[i][0,:].tolist()
        plan.drvar[i].y_cp = drvars[i][1,:].tolist()
        plan.drvar[i].z_cp = drvars[i][2,:].tolist()

        plan.hvar[i].x_cp = hvars[i][0,:].tolist()
        plan.dhvar[i].x_cp = dhvars[i][0,:].tolist()

        plan.ids[i] = idvars[i]
        plan.other_names[i] = other_names[i]
    return plan
    

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(SetVerbosePlan, 'set_plan')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetVerbosePlan.Request()
    
    def send_request(self, rvars, hvars, idvars, other_names, orvars=None, ohvars=None, oidvars=None, oother_names=None):
        """
        Sends a request to the service to set the plan.
        rvars: list of numpy arrays of shape (3, n) where n is the number of control points
        hvars: list of numpy arrays of shape (1, n) where n is the number of control points
        idvars: list of strings
        other_names: list of strings
        returns: response from the service
        """
        assert(len(rvars) == len(hvars))
        plan = plan_to_plan_msg(rvars,hvars,idvars,other_names)
        
        if orvars is not None:
            assert(len(orvars) == len(ohvars))
            object_plan = plan_to_plan_msg(orvars,ohvars,oidvars,oother_names)
            self.req.replanned = True
            self.req.object_plan = object_plan
        else:
            self.req.replanned = False

        self.req.plan = plan
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()



class MainPlanner(Node):
    def __init__(self, node):
        super().__init__('simple_planner')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.minimal_client = MinimalClientAsync()
        self.node = node

        self.robot_name = self.get_namespace()
        self.scenario_name = self.declare_parameter('scenario_name','throw_and_catch').value

        # Subscribers to the state
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            'fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile)

        # now we can send the request if the subscriber to "plan" receives a True message
        new_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.compute_plan_sub = self.create_subscription(
            StampedBool,
            'impact_stl/compute_plan',
            self.compute_plan_callback,
            new_qos_profile)

        
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.init_position = np.array([0.0, 0.0, 0.0])
        self.init_velocity = np.array([0.0, 0.0, 0.0])
        self.get_logger().info('SimplePlanner initialized')        
        
    def vehicle_local_position_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_local_position[0] = msg.y
        self.vehicle_local_position[1] = msg.x
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vy
        self.vehicle_local_velocity[1] = msg.vx
        self.vehicle_local_velocity[2] = -msg.vz

    def compute_plan_callback(self, msg):
        self.get_logger().info('Computing plan')
        package_share_directory = get_package_share_directory('inter_stl')
        plans_path = os.path.join(package_share_directory)

        # list of np.arrays, or strings one for each bezier curve. Each curve is of shape (d, n) where n is the number of control points
        # d=3 for rvars, d=1 for hvars
        self.rvars,self.hvars,self.idvars,self.other_names = csv_to_plan(robot_name=self.robot_name,
                                                                         scenario_name=self.scenario_name,
                                                                         path=plans_path)                      

        # right now we set the init position so we can reset back to it!
        self.init_position = self.vehicle_local_position
        self.get_logger().info('Plan computed')

        if msg.data:
            self.get_logger().info('Sending plan')
            self.minimal_client.send_request(self.rvars, self.hvars, self.idvars, self.other_names)
            self.get_logger().info('Plan received')




def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('main_planner')
    simple_planner = MainPlanner(node)
    rclpy.spin(simple_planner)

    simple_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    