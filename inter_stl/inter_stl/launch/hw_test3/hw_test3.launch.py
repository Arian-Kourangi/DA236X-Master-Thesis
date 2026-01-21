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
            parameters=[{'x0':0.5, 'y0':-1.25, 'z0':0.0, 'vx0':0.0, 'vy0':0.0, 'vz0':0.0},
                        {'scenario_name':'hw_test3'},
                        {'object_ns':'/snap'},
                        {'enable_cbf':True},
                        {'hw':HW}] # NOTE: This must be True to use mocap/PX4 data for object
        ),
        Node(
           package='inter_stl',
           namespace='crackle',
           executable='ff_rate_mpc', # spacecraft_mpc, spacecraft_impact_mpc
           name='crackle_mpc',
           output='screen',
           emulate_tty=True,
           parameters=[{'x0':3.25, 'y0':1.5, 'z0':0.0, 'vx0':0.0, 'vy0':0.0, 'vz0':0.0},
                       {'scenario_name':'hw_test3'},
                       {'object_ns':'/snap'},
                       {'enable_cbf':True},
                       {'hw':HW}] # NOTE: This must be True to use mocap/PX4 data for object
        ),

        # Bezier planner
        Node(
            package='inter_stl',
            namespace='pop',
            executable='main_planner',
            name='pop_planner',
            output='screen',
            emulate_tty=True,
            parameters=[{'scenario_name':'hw_test3'}]
        ),
        Node(
           package='inter_stl',
           namespace='crackle',
           executable='main_planner',
           name='crackle_planner',
           output='screen',
           emulate_tty=True,
           parameters=[{'scenario_name':'hw_test3'}]
        ),

        # No replanner for now since we don't have an object to move
        # Replanner
        #    
        Node(
            package='inter_stl',
            namespace='pop',
            executable='replanner',
            name='pop_replanner',
            output='screen',
            parameters=[{'object_ns':'/snap'},
                        {'scenario_name':'hw_test3'},
                        {'hw':HW}]
        ),
        # Replanner
        Node(
           package='inter_stl',
           namespace='crackle',
           executable='replanner',
           name='crackle_replanner',
           output='screen',
           parameters=[{'object_ns':'/snap'},
                       {'scenario_name':'hw_test3'},
                       {'hw':HW}]
        ),
        Node(
            package='inter_stl',
            namespace='snap',
            executable='ff_rate_mpc_velocity_keeping', 
            name='snap_velocity_keeping_mpc',
            output='screen',
            emulate_tty=True,
            parameters=[{'x0':1.25, 'y0':-0.5, 'z0':0.0, 'vx0':0.0, 'vy0':0.0, 'vz0':0.0},
                        {'hw':HW}]
        ),

    ])
