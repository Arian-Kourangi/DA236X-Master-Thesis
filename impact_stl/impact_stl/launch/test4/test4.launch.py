#!/usr/bin/env python
__author__ = "Arian Kourangi"
__contact__ = "arianke@kth.se"

from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    
    return LaunchDescription([
        # MPC controller
        Node(
            package='impact_stl',
            namespace='snap',
            executable='ff_rate_mpc_impact', # spacecraft_mpc, spacecraft_impact_mpc
            name='snap_mpc',
            output='screen',
            emulate_tty=True,
            parameters=[{'x0':1.0, 'y0':4.0, 'z0':0.0, 'vx0':0.0, 'vy0':0.0, 'vz0':0.0},
                        {'scenario_name':'test4'},
                        {'object_ns':'/pop'},
                        {'enable_cbf':True}]
        ),
        Node(
            package='impact_stl',
            namespace='crackle',
            executable='ff_rate_mpc_impact', # spacecraft_mpc, spacecraft_impact_mpc
            name='crackle_mpc',
            # output='screen',
            emulate_tty=True,
            parameters=[{'x0':2.0, 'y0':9.0, 'z0':0.0, 'vx0':0.0, 'vy0':0.0, 'vz0':0.0},
                        {'scenario_name':'test4'},
                        {'object_ns':'/pop'},
                        {'enable_cbf':True}]
        ),

        # Bezier planner
        Node(
            package='impact_stl',
            namespace='snap',
            executable='main_planner',
            name='snap_planner',
            output='screen',
            emulate_tty=True,
            parameters=[{'scenario_name':'test4'}]
        ),
        Node(
            package='impact_stl',
            namespace='crackle',
            executable='main_planner',
            name='crackle_planner',
            # output='screen',
            emulate_tty=True,
            parameters=[{'scenario_name':'test4'}]
        ),

        # Replanner
        Node(
            package='impact_stl',
            namespace='snap',
            executable='replanner',
            name='snap_replanner',
            output='screen',
            parameters=[{'object_ns':'/pop'},
                        {'scenario_name':'test4'}]
        ),
        # Replanner
        Node(
            package='impact_stl',
            namespace='crackle',
            executable='replanner',
            name='crackle_replanner',
            # output='screen',
            parameters=[{'object_ns':'/pop'},
                        {'scenario_name':'test4'}]
        ),

    ])
