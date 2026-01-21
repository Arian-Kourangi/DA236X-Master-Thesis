#!/usr/bin/env python
__author__ = "Arian Kourangi"
__contact__ = "arianke@kth.se"

from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
import os

HW = True
def generate_launch_description():
    
    return LaunchDescription([
        # MPC controller
        Node(
            package='inter_stl',
            namespace='pop',
            executable='ff_rate_mpc', # spacecraft_mpc, spacecraft_impact_mpc
            name='pop_mpc',
            output='screen',
            emulate_tty=True,
            parameters=[{'x0':0.5, 'y0':0.0, 'z0':0.0, 'vx0':0.0, 'vy0':0.0, 'vz0':0.0},
                        {'scenario_name':'hw_test2'},
                        {'object_ns':'/crackle'},
                        {'enable_cbf':True},
                        {'hw':HW}] # NOTE: This must be True to use mocap/PX4 data for object
        ),
        #Node(
        #    package='inter_stl',
        #    namespace='crackle',
        #    executable='ff_rate_mpc_impact', # spacecraft_mpc, spacecraft_impact_mpc
        #    name='crackle_mpc',
        #    # output='screen',
        #    emulate_tty=True,
        #    parameters=[{'x0':5.0, 'y0':20.0, 'z0':0.0, 'vx0':0.0, 'vy0':0.0, 'vz0':0.0},
        #                {'scenario_name':'hw_test'},
        #                {'object_ns':'/pop'},
        #                {'enable_cbf':True},
        #                {'hw':True}] # NOTE: This must be True to use mocap/PX4 data for object
        #),

        # Bezier planner
        Node(
            package='inter_stl',
            namespace='pop',
            executable='main_planner',
            name='pop_planner',
            output='screen',
            emulate_tty=True,
            parameters=[{'scenario_name':'hw_test2'}]
        ),
        #Node(
        #    package='inter_stl',
        #    namespace='crackle',
        #    executable='main_planner',
        #    name='crackle_planner',
        #    # output='screen',
        #    emulate_tty=True,
        #    parameters=[{'scenario_name':'test1'}]
        #),

        # No replanner for now since we don't have an object to move
        # Replanner
        #    
        Node(
            package='inter_stl',
            namespace='pop',
            executable='replanner',
            name='pop_replanner',
            output='screen',
            parameters=[{'object_ns':'/crackle'},
                        {'scenario_name':'hw_test2'},
                        {'hw':HW}]
        ),
        # Replanner
        #Node(
        #    package='inter_stl',
        #    namespace='crackle',
        #    executable='replanner',
        #    name='crackle_replanner',
        #    # output='screen',
        #    parameters=[{'object_ns':'/pop'},
        #                {'scenario_name':'hw_test'},
        #                {'hw':True}]
        #),
        # Velocity keeping MPC controller for object
        Node(
            package='inter_stl',
            namespace='crackle',
            executable='ff_rate_mpc_velocity_keeping', 
            name='snap_velocity_keeping_mpc',
            output='screen',
            emulate_tty=True,
            parameters=[{'x0':1.5, 'y0':0.0, 'z0':0.0, 'vx0':0.0, 'vy0':0.0, 'vz0':0.0},
                        {'hw':HW}]
        ),
    ])
